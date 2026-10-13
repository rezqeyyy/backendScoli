package main

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
)

// AxisNode adalah satu titik pada sumbu tubuh: keypoint ("k") atau titik tengah pasangan ("p").
// Di pipeline_config.json bentuknya ["k", 3] atau ["p", 0].
type AxisNode struct {
	Kind  string
	Index int
}

func (a *AxisNode) UnmarshalJSON(b []byte) error {
	var raw []json.RawMessage
	if err := json.Unmarshal(b, &raw); err != nil || len(raw) != 2 {
		return fmt.Errorf("node sumbu harus berbentuk [jenis, indeks]: %s", string(b))
	}
	if err := json.Unmarshal(raw[0], &a.Kind); err != nil {
		return err
	}
	return json.Unmarshal(raw[1], &a.Index)
}

type Schema struct {
	Pairs   [][]int    `json:"pairs"`
	Midline []int      `json:"midline"`
	Axis    []AxisNode `json:"axis"`
}

type InputGuard struct {
	MinSide          int     `json:"min_side"`
	RequireGrayscale bool    `json:"require_grayscale"`
	MaxSaturation    float64 `json:"max_saturation"`
	MinUniqueGray    int     `json:"min_unique_gray"`
	MinEdgeStd       float64 `json:"min_edge_std"`
}

// Background: pengaturan hapus background dengan YOLOv8-seg. Tidak ada di config lama (= mati).
type Background struct {
	Enabled     bool    `json:"enabled"`
	SegConf     float64 `json:"seg_conf"`
	PersonClass int     `json:"person_class"`
	BgColor     int     `json:"bg_color"`
	MaskDilate  int     `json:"mask_dilate"`
	MinMaskFrac float64 `json:"min_mask_frac"`
}

// Calibration: temperature scaling hasil Bagian kalibrasi di notebook.
// Probabilitas dilunakkan dengan p_i ∝ p_i^(1/T); T > 1 membuat model tidak terlalu yakin.
// Nilai 0 atau tidak ada = 1 (tanpa kalibrasi), supaya config lama tetap jalan.
type Calibration struct {
	MLPTemperature float64 `json:"mlp_temperature"`
	CNNTemperature float64 `json:"cnn_temperature"`
}

// CNNConfig: pengaturan potongan gambar untuk classifier ResNet.
type CNNConfig struct {
	Backbone string  `json:"backbone"`
	ImgSize  int     `json:"img_size"`
	CropPad  float64 `json:"crop_pad"`
}

// Config adalah isi pipeline_config.json yang diekspor notebook (Bagian 14).
type Config struct {
	ClassNames      []string    `json:"class_names"`
	NormalClass     string      `json:"normal_class"`
	RiskThresholds  []float64   `json:"risk_thresholds"`
	RiskLevels      []string    `json:"risk_levels"`
	Disclaimer      string      `json:"disclaimer"`
	ImgSize         int         `json:"img_size"`
	StretchToSquare bool        `json:"stretch_to_square"`
	DetConf         float64     `json:"det_conf"`
	KptConf         float64     `json:"kpt_conf"`
	MinVisibleKpts  int         `json:"min_visible_kpts"`
	ReviewThreshold float64     `json:"review_threshold"`
	KptShape        []int       `json:"kpt_shape"`
	KptNames        []string    `json:"kpt_names"`
	Schema          Schema      `json:"schema"`
	PairNames       []string    `json:"pair_names"`
	AxisNames       []string    `json:"axis_names"`
	FeatureNames    []string    `json:"feature_names"`
	DisplayFeatures []string    `json:"display_features"`
	InputGuard      InputGuard  `json:"input_guard"`
	FinalClassifier string      `json:"final_classifier"` // mlp | resnet | ensemble (kosong = mlp)
	Background      Background  `json:"background"`
	CNN             CNNConfig   `json:"cnn"`
	Calibration     Calibration `json:"calibration"`
	// DetectOnOriginal: bila deteksi pada gambar tanpa background gagal atau kurang jelas,
	// coba lagi pada gambar asli. Tidak ada di config = aktif.
	DetectOnOriginal *bool `json:"detect_on_original"`
}

// UsesCNN: apakah classifier akhir membutuhkan scoliosis_cnn.onnx.
func (c *Config) UsesCNN() bool {
	return c.FinalClassifier == "resnet" || c.FinalClassifier == "ensemble"
}

func (c *Config) NumKpts() int { return c.KptShape[0] }
func (c *Config) KptDim() int  { return c.KptShape[1] }

func LoadConfig(path string) (*Config, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var c Config
	if err := json.Unmarshal(b, &c); err != nil {
		return nil, fmt.Errorf("%s tidak valid: %w", path, err)
	}
	if err := c.applyEnvOverrides(); err != nil {
		return nil, err
	}
	if err := c.validate(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return &c, nil
}

func (c *Config) validate() error {
	if len(c.ClassNames) == 0 {
		return fmt.Errorf("class_names kosong")
	}
	if len(c.KptShape) != 2 || c.KptShape[0] < 1 || (c.KptShape[1] != 2 && c.KptShape[1] != 3) {
		return fmt.Errorf("kpt_shape harus [jumlah keypoint, 2 atau 3]")
	}
	k := c.NumKpts()
	if len(c.KptNames) != k {
		return fmt.Errorf("kpt_names harus berisi %d nama", k)
	}
	if c.ImgSize <= 0 {
		return fmt.Errorf("img_size tidak valid")
	}
	if !c.StretchToSquare {
		return fmt.Errorf("stretch_to_square=false (letterbox) belum didukung backend ini")
	}
	if len(c.RiskThresholds) != 2 || len(c.RiskLevels) != 3 {
		return fmt.Errorf("risk_thresholds harus 2 angka dan risk_levels harus 3 nama")
	}
	if c.RiskThresholds[0] >= c.RiskThresholds[1] {
		return fmt.Errorf("risk_thresholds harus naik, misalnya [0.35, 0.65]")
	}
	if c.DetConf <= 0 || c.DetConf >= 1 || c.KptConf < 0 || c.KptConf >= 1 {
		return fmt.Errorf("det_conf harus di antara 0 dan 1, kpt_conf di antara 0 dan 1")
	}
	if c.MinVisibleKpts < 1 || c.MinVisibleKpts > k {
		return fmt.Errorf("min_visible_kpts harus 1 sampai %d", k)
	}
	if len(c.PairNames) != len(c.Schema.Pairs) || len(c.AxisNames) != len(c.Schema.Axis) {
		return fmt.Errorf("pair_names/axis_names tidak cocok dengan schema")
	}
	switch c.FinalClassifier {
	case "":
		c.FinalClassifier = "mlp" // config dari notebook versi pertama
	case "mlp", "resnet", "ensemble":
	default:
		return fmt.Errorf("final_classifier tidak dikenal: %q", c.FinalClassifier)
	}
	if c.UsesCNN() && c.CNN.ImgSize <= 0 {
		return fmt.Errorf("cnn.img_size tidak valid")
	}
	if c.Background.Enabled && (c.Background.BgColor < 0 || c.Background.BgColor > 255 || c.Background.PersonClass < 0) {
		return fmt.Errorf("pengaturan background tidak valid")
	}
	for _, p := range c.Schema.Pairs {
		if len(p) != 2 || p[0] < 0 || p[0] >= k || p[1] < 0 || p[1] >= k {
			return fmt.Errorf("schema.pairs berisi indeks keypoint yang tidak valid")
		}
	}
	for _, n := range c.Schema.Axis {
		switch n.Kind {
		case "k":
			if n.Index < 0 || n.Index >= k {
				return fmt.Errorf("schema.axis berisi indeks keypoint yang tidak valid")
			}
		case "p":
			if n.Index < 0 || n.Index >= len(c.Schema.Pairs) {
				return fmt.Errorf("schema.axis berisi indeks pasangan yang tidak valid")
			}
		default:
			return fmt.Errorf("jenis node sumbu tidak dikenal: %q", n.Kind)
		}
	}
	// Urutan fitur di Go harus persis sama dengan Python, karena MLP dilatih dengan urutan itu.
	names, _ := c.PostureFeatures(make([][3]float64, k))
	if len(names) != len(c.FeatureNames) {
		return fmt.Errorf("jumlah fitur Go (%d) tidak sama dengan feature_names (%d)", len(names), len(c.FeatureNames))
	}
	for i := range names {
		if names[i] != c.FeatureNames[i] {
			return fmt.Errorf("fitur ke-%d berbeda: Go %q, config %q", i, names[i], c.FeatureNames[i])
		}
	}
	return nil
}

// applyEnvOverrides: ambang bisa diubah lewat environment variable (misalnya di dashboard Render)
// tanpa mengekspor ulang model. Kosong = pakai nilai dari pipeline_config.json.
//
//	DET_CONF, KPT_CONF, MIN_VISIBLE_KPTS, MLP_TEMPERATURE, CNN_TEMPERATURE,
//	RISK_THRESHOLDS ("0.35,0.65"), DETECT_ON_ORIGINAL ("true"/"false")
func (c *Config) applyEnvOverrides() error {
	num := func(key string, dst *float64) error {
		if v := os.Getenv(key); v != "" {
			f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
			if err != nil {
				return fmt.Errorf("%s bukan angka: %q", key, v)
			}
			*dst = f
		}
		return nil
	}
	for key, dst := range map[string]*float64{
		"DET_CONF":        &c.DetConf,
		"KPT_CONF":        &c.KptConf,
		"MLP_TEMPERATURE": &c.Calibration.MLPTemperature,
		"CNN_TEMPERATURE": &c.Calibration.CNNTemperature,
	} {
		if err := num(key, dst); err != nil {
			return err
		}
	}
	if v := os.Getenv("MIN_VISIBLE_KPTS"); v != "" {
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil || n < 1 {
			return fmt.Errorf("MIN_VISIBLE_KPTS harus bilangan bulat >= 1: %q", v)
		}
		c.MinVisibleKpts = n
	}
	if v := os.Getenv("RISK_THRESHOLDS"); v != "" {
		parts := strings.Split(v, ",")
		if len(parts) != 2 {
			return fmt.Errorf("RISK_THRESHOLDS harus dua angka dipisah koma: %q", v)
		}
		t := make([]float64, 2)
		for i, p := range parts {
			f, err := strconv.ParseFloat(strings.TrimSpace(p), 64)
			if err != nil {
				return fmt.Errorf("RISK_THRESHOLDS tidak valid: %q", v)
			}
			t[i] = f
		}
		c.RiskThresholds = t
	}
	if v := os.Getenv("DETECT_ON_ORIGINAL"); v != "" {
		b, err := strconv.ParseBool(strings.TrimSpace(v))
		if err != nil {
			return fmt.Errorf("DETECT_ON_ORIGINAL harus true/false: %q", v)
		}
		c.DetectOnOriginal = &b
	}
	return nil
}

// detectOnOriginal: nilai bawaan aktif.
func (c *Config) detectOnOriginal() bool {
	return c.DetectOnOriginal == nil || *c.DetectOnOriginal
}

// temperature: 0 atau negatif dianggap 1 (tanpa kalibrasi).
func temperature(t float64) float64 {
	if t <= 0 {
		return 1
	}
	return t
}

// calibrate menerapkan temperature scaling pada probabilitas: softmax(log(p) / T).
// Hasilnya sama persis dengan softmax(logit / T), karena log-softmax hanya bergeser konstanta.
func calibrate(probs []float64, t float64) []float64 {
	t = temperature(t)
	out := make([]float64, len(probs))
	if t == 1 {
		copy(out, probs)
		return out
	}
	maxL := math.Inf(-1)
	for i, p := range probs {
		out[i] = math.Log(math.Max(p, 1e-7)) / t
		maxL = math.Max(maxL, out[i])
	}
	sum := 0.0
	for i := range out {
		out[i] = math.Exp(out[i] - maxL)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}