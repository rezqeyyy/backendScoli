package main

import (
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"

	"github.com/gofiber/fiber/v2"
)

var (
	cfg    *Config
	engine *Engine
	// Membatasi jumlah gambar yang dianalisis bersamaan supaya pemakaian memori terjaga
	// di server kecil (misalnya 512 MB). Permintaan lain menunggu giliran.
	slots chan struct{}
)

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func defaultLibPath() string {
	switch runtime.GOOS {
	case "windows":
		return "onnxruntime.dll"
	case "darwin":
		return "libonnxruntime.dylib"
	default:
		return "libonnxruntime.so"
	}
}

func main() {
	modelsDir := envOr("MODELS_DIR", "models")
	libPath := envOr("ONNXRUNTIME_LIB", defaultLibPath())
	port := envOr("PORT", "3000")
	maxConcurrent, err := strconv.Atoi(envOr("MAX_CONCURRENT", "2"))
	if err != nil || maxConcurrent < 1 {
		maxConcurrent = 2
	}
	slots = make(chan struct{}, maxConcurrent)

	if cfg, err = LoadConfig(filepath.Join(modelsDir, "pipeline_config.json")); err != nil {
		log.Fatalf("Gagal memuat konfigurasi: %v", err)
	}
	if engine, err = NewEngine(cfg, libPath, modelsDir); err != nil {
		log.Fatalf("Gagal menyiapkan model: %v", err)
	}
	defer engine.Close()
	log.Printf("Model siap: %d kelas %v, %d keypoint, %d fitur, classifier %s, hapus background %v",
		len(cfg.ClassNames), cfg.ClassNames, cfg.NumKpts(), len(cfg.FeatureNames), cfg.FinalClassifier, cfg.Background.Enabled)

	// Batas unggahan bawaan Fiber hanya 4 MB; foto kamera ponsel sering lebih besar.
	app := fiber.New(fiber.Config{BodyLimit: 20 * 1024 * 1024})
	// Pemeriksaan kesehatan: dijawab di beberapa path umum supaya lolos apa pun
	// isi "Health Check Path" di platform hosting.
	health := func(c *fiber.Ctx) error { return c.JSON(fiber.Map{"status": "ok"}) }
	for _, p := range []string{"/", "/health", "/healthz"} {
		app.Get(p, health)
	}
	app.Post("/api/v1/detect-scoliosis", handleDetection)

	fmt.Printf("Server berjalan di port %s\n", port)
	log.Fatal(app.Listen(":" + port))
}

func handleDetection(c *fiber.Ctx) error {
	file, err := c.FormFile("image")
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"status": "error", "message": "Gambar tidak ditemukan di form-data (field: image)."})
	}
	f, err := file.Open()
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"status": "error", "message": "Gagal membuka gambar."})
	}
	defer f.Close()
	data, err := io.ReadAll(f)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"status": "error", "message": "Gagal membaca gambar."})
	}

	slots <- struct{}{}
	result, err := Analyze(data)
	<-slots
	if err != nil {
		log.Printf("analisis gagal: %v", err)
		return c.Status(500).JSON(fiber.Map{"status": "error", "message": "Terjadi kesalahan saat menganalisis gambar."})
	}
	return c.JSON(result)
}

// Keypoint: koordinat 0-1 relatif terhadap lebar dan tinggi gambar.
type Keypoint struct {
	Name string  `json:"name"`
	X    float64 `json:"x"`
	Y    float64 `json:"y"`
	Conf float64 `json:"conf"`
}

// Overlay: bahan untuk menggambar titik pengukuran di atas foto pada aplikasi.
// Garis pasangan menghubungkan dua keypoint (misalnya bahu kiri-kanan); sumbu tubuh adalah
// garis dari atas ke bawah yang dipakai menghitung kemiringan dan simpangan.
type Overlay struct {
	Pairs [][2]int     `json:"pairs"` // indeks keypoint yang dihubungkan
	Axis  [][2]float64 `json:"axis"`  // titik sumbu tubuh berurutan atas -> bawah, koordinat 0-1
}

// Result adalah respons untuk aplikasi mobile; sama dengan keluaran analyze() di notebook,
// ditambah `overlay`.
type Result struct {
	Status          string             `json:"status"` // ok | invalid_image | no_detection | low_quality
	Message         string             `json:"message,omitempty"`
	Label           string             `json:"label,omitempty"`
	RiskScore       *float64           `json:"risk_score,omitempty"`
	RiskLevel       string             `json:"risk_level,omitempty"`
	Confidence      *float64           `json:"confidence,omitempty"`
	NeedsReview     *bool              `json:"needs_review,omitempty"`
	Probabilities   map[string]float64 `json:"probabilities,omitempty"`
	PostureFeatures map[string]float64 `json:"posture_features,omitempty"`
	Keypoints       []Keypoint         `json:"keypoints,omitempty"`
	BBox            []float64          `json:"bbox,omitempty"`
	Overlay         *Overlay           `json:"overlay,omitempty"`
	Classifier      string             `json:"classifier,omitempty"`
	BgRemoved       *bool              `json:"background_removed,omitempty"`
	Disclaimer      string             `json:"disclaimer,omitempty"`
}

func round(v float64, digits int) float64 {
	p := math.Pow(10, float64(digits))
	return math.Round(v*p) / p
}

// riskFromProbs: skor risiko = 1 - P(kelas normal), lalu dibagi tiga tingkat.
func riskFromProbs(probs []float64) (*float64, string) {
	for i, name := range cfg.ClassNames {
		if strings.EqualFold(name, cfg.NormalClass) {
			score := 1 - probs[i]
			level := 0
			if score >= cfg.RiskThresholds[0] {
				level++
			}
			if score >= cfg.RiskThresholds[1] {
				level++
			}
			score = round(score, 3)
			return &score, cfg.RiskLevels[level]
		}
	}
	return nil, "Tidak diketahui"
}

// Analyze: gambar -> validasi -> hapus background -> YOLOv8 Pose -> fitur postur dan/atau
// potongan gambar -> classifier akhir -> hasil. Sama dengan analyze() di notebook.
func Analyze(data []byte) (*Result, error) {
	img, origW, origH, err := decodeImage(data)
	if err != nil {
		return &Result{Status: "invalid_image", Message: "Gambar tidak bisa dibaca. Gunakan JPEG atau PNG."}, nil
	}
	if ok, reason := validateInput(img, origW, origH, cfg.InputGuard); !ok {
		return &Result{Status: "invalid_image", Message: reason}, nil
	}

	// Gambar kerja 640x640; background dihapus di sini sebelum model lain melihatnya.
	work := stretch(img, cfg.ImgSize)
	bgRemoved, err := engine.RemoveBackground(work)
	if err != nil {
		return nil, err
	}
	det, err := engine.Detect(toTensor(work))
	if err != nil {
		return nil, err
	}
	if det == nil {
		return &Result{Status: "no_detection", Message: "Punggung tidak terdeteksi dengan yakin. Ambil ulang foto."}, nil
	}
	visible := 0
	for _, k := range det.Kpts {
		if k[2] > cfg.KptConf {
			visible++
		}
	}
	if visible < cfg.MinVisibleKpts {
		return &Result{Status: "low_quality", Message: fmt.Sprintf("Hanya %d dari %d titik tubuh yang terlihat jelas. Ambil ulang foto.", visible, len(det.Kpts))}, nil
	}

	names, feats := cfg.PostureFeatures(det.Kpts)
	probs, err := engine.Classify(feats)
	if err != nil {
		return nil, err
	}
	if cfg.UsesCNN() {
		crop := cropForCNN(work, det.Box, cfg.CNN.CropPad, cfg.CNN.ImgSize, cfg.Background.BgColor)
		pCNN, err := engine.ClassifyImage(toTensor(crop))
		if err != nil {
			return nil, err
		}
		for i := range probs {
			if cfg.FinalClassifier == "resnet" {
				probs[i] = pCNN[i]
			} else { // ensemble: rata-rata MLP dan CNN
				probs[i] = (probs[i] + pCNN[i]) / 2
			}
		}
	}
	best := 0
	for i, p := range probs {
		if p > probs[best] {
			best = i
		}
	}

	size := float64(cfg.ImgSize) // gambar di-stretch, jadi pecahan ini berlaku juga untuk gambar asli
	res := &Result{
		Status:          "ok",
		Label:           cfg.ClassNames[best],
		Probabilities:   map[string]float64{},
		PostureFeatures: map[string]float64{},
		Classifier:      cfg.FinalClassifier,
		BgRemoved:       &bgRemoved,
		Disclaimer:      cfg.Disclaimer,
	}
	res.RiskScore, res.RiskLevel = riskFromProbs(probs)
	conf := round(probs[best], 3)
	review := probs[best] < cfg.ReviewThreshold
	res.Confidence, res.NeedsReview = &conf, &review
	for i, p := range probs {
		res.Probabilities[cfg.ClassNames[i]] = round(p, 3)
	}
	display := map[string]bool{}
	for _, n := range cfg.DisplayFeatures {
		display[n] = true
	}
	for i, n := range names {
		if display[n] {
			res.PostureFeatures[n] = round(feats[i], 2)
		}
	}
	for i, k := range det.Kpts {
		res.Keypoints = append(res.Keypoints, Keypoint{Name: cfg.KptNames[i], X: round(k[0]/size, 4), Y: round(k[1]/size, 4), Conf: round(k[2], 3)})
	}
	for _, v := range det.Box {
		res.BBox = append(res.BBox, round(v/size, 4))
	}
	// Garis bantu untuk overlay: pasangan kiri-kanan dan sumbu tubuh, dalam koordinat 0-1.
	norm := make([][2]float64, len(det.Kpts))
	for i, k := range det.Kpts {
		norm[i] = [2]float64{k[0] / size, k[1] / size}
	}
	res.Overlay = &Overlay{Pairs: [][2]int{}, Axis: [][2]float64{}}
	for _, p := range cfg.Schema.Pairs {
		res.Overlay.Pairs = append(res.Overlay.Pairs, [2]int{p[0], p[1]})
	}
	for _, a := range cfg.axisPoints(norm) {
		res.Overlay.Axis = append(res.Overlay.Axis, [2]float64{round(a[0], 4), round(a[1], 4)})
	}
	return res, nil
}