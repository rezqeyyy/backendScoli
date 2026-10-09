package main

import (
	"fmt"
	"log"
	"math"
	"path/filepath"
	"strings"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

// Engine memegang dua sesi ONNX Runtime: YOLOv8 Pose dan classifier MLP.
type Engine struct {
	cfg      *Config
	pose     *ort.DynamicAdvancedSession
	mlp      *ort.DynamicAdvancedSession
	poseRows int // 4 + jumlah kelas + keypoint x dimensi
	mu       sync.Mutex
}

// Detection adalah kandidat terbaik dari YOLOv8 Pose (koordinat piksel pada gambar img_size x img_size).
type Detection struct {
	Kpts  [][3]float64 // x, y, conf
	Box   [4]float64   // x1, y1, x2, y2
	Class int
	Conf  float64
}

func NewEngine(cfg *Config, libPath, modelsDir string) (*Engine, error) {
	ort.SetSharedLibraryPath(libPath)
	if err := ort.InitializeEnvironment(); err != nil {
		hint := ""
		if strings.Contains(err.Error(), "not a valid Win32") {
			hint = " (arsitektur onnxruntime.dll tidak cocok dengan hasil `go env GOARCH`)"
		}
		return nil, fmt.Errorf("gagal memuat ONNX Runtime dari %q%s: %w", libPath, hint, err)
	}
	// ONNX Runtime bisa mengirim data pemakaian ke Microsoft; tidak diperlukan di sini.
	if err := ort.DisableTelemetry(); err != nil {
		log.Printf("telemetri ONNX Runtime tidak bisa dimatikan: %v", err)
	}

	e := &Engine{cfg: cfg, poseRows: 4 + len(cfg.ClassNames) + cfg.NumKpts()*cfg.KptDim()}
	var err error
	posePath := filepath.Join(modelsDir, "scoliosis_pose.onnx")
	mlpPath := filepath.Join(modelsDir, "scoliosis_mlp.onnx")

	// Cek bentuk input/output dulu, supaya model yang tidak cocok dengan config ketahuan saat start.
	in, out, err := ort.GetInputOutputInfo(posePath)
	if err != nil {
		return nil, fmt.Errorf("gagal membaca %s: %w", posePath, err)
	}
	if len(in) != 1 || len(out) < 1 {
		return nil, fmt.Errorf("%s: jumlah input/output tidak sesuai model YOLOv8 Pose", posePath)
	}
	if d := out[0].Dimensions; len(d) != 3 || (d[1] > 0 && int(d[1]) != e.poseRows) {
		return nil, fmt.Errorf("%s: bentuk output %v, seharusnya [1 %d N] sesuai kelas dan keypoint di config", posePath, d, e.poseRows)
	}
	if e.pose, err = ort.NewDynamicAdvancedSession(posePath, []string{in[0].Name}, []string{out[0].Name}, nil); err != nil {
		return nil, fmt.Errorf("gagal memuat %s: %w", posePath, err)
	}

	in, out, err = ort.GetInputOutputInfo(mlpPath)
	if err != nil {
		return nil, fmt.Errorf("gagal membaca %s: %w", mlpPath, err)
	}
	if len(in) != 1 || len(out) != 1 {
		return nil, fmt.Errorf("%s: jumlah input/output tidak sesuai classifier", mlpPath)
	}
	if d := in[0].Dimensions; len(d) != 2 || int(d[1]) != len(cfg.FeatureNames) {
		return nil, fmt.Errorf("%s: bentuk input %v, seharusnya [N %d] sesuai feature_names", mlpPath, d, len(cfg.FeatureNames))
	}
	if e.mlp, err = ort.NewDynamicAdvancedSession(mlpPath, []string{in[0].Name}, []string{out[0].Name}, nil); err != nil {
		return nil, fmt.Errorf("gagal memuat %s: %w", mlpPath, err)
	}
	return e, nil
}

func (e *Engine) Close() {
	if e.pose != nil {
		e.pose.Destroy()
	}
	if e.mlp != nil {
		e.mlp.Destroy()
	}
	ort.DestroyEnvironment()
}

// run menjalankan satu sesi dengan satu input float32 dan mengembalikan data serta bentuk outputnya.
func (e *Engine) run(s *ort.DynamicAdvancedSession, shape ort.Shape, data []float32) ([]float32, ort.Shape, error) {
	in, err := ort.NewTensor(shape, data)
	if err != nil {
		return nil, nil, err
	}
	defer in.Destroy()

	outs := []ort.Value{nil}
	e.mu.Lock()
	err = s.Run([]ort.Value{in}, outs)
	e.mu.Unlock()
	if err != nil {
		return nil, nil, err
	}
	defer outs[0].Destroy()

	t, ok := outs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, nil, fmt.Errorf("output model bukan tensor float32")
	}
	res := make([]float32, len(t.GetData()))
	copy(res, t.GetData()) // salin sebelum tensor dibebaskan
	return res, t.GetShape(), nil
}

// Detect menjalankan YOLOv8 Pose dan mengambil kandidat dengan skor kelas tertinggi.
// Kandidat teratas selalu lolos NMS, jadi NMS tidak perlu dijalankan untuk mengambil satu deteksi terbaik.
// Mengembalikan nil bila skor terbaik di bawah det_conf.
func (e *Engine) Detect(tensor []float32) (*Detection, error) {
	s := int64(e.cfg.ImgSize)
	out, shape, err := e.run(e.pose, ort.NewShape(1, 3, s, s), tensor)
	if err != nil {
		return nil, err
	}
	if len(shape) != 3 || int(shape[1]) != e.poseRows {
		return nil, fmt.Errorf("bentuk output pose tidak terduga: %v", shape)
	}
	// Tata letak: out[baris*n + i]. Baris 0-3 = cx, cy, w, h; lalu skor tiap kelas; lalu keypoint.
	n := int(shape[2])
	nc, k, kd := len(e.cfg.ClassNames), e.cfg.NumKpts(), e.cfg.KptDim()

	best, bestCls, bestConf := -1, 0, float32(math.Inf(-1))
	for c := 0; c < nc; c++ {
		row := out[(4+c)*n : (5+c)*n]
		for i, v := range row {
			if v > bestConf {
				best, bestCls, bestConf = i, c, v
			}
		}
	}
	if best < 0 || float64(bestConf) < e.cfg.DetConf {
		return nil, nil
	}

	at := func(row int) float64 { return float64(out[row*n+best]) }
	cx, cy, w, h := at(0), at(1), at(2), at(3)
	det := &Detection{
		Kpts:  make([][3]float64, k),
		Box:   [4]float64{cx - w/2, cy - h/2, cx + w/2, cy + h/2},
		Class: bestCls,
		Conf:  float64(bestConf),
	}
	for j := 0; j < k; j++ {
		base := 4 + nc + j*kd
		det.Kpts[j] = [3]float64{at(base), at(base + 1), 1}
		if kd == 3 {
			det.Kpts[j][2] = at(base + 2)
		}
	}
	return det, nil
}

// Classify mengirim vektor fitur ke MLP dan mengembalikan probabilitas tiap kelas.
func (e *Engine) Classify(features []float64) ([]float64, error) {
	f32 := make([]float32, len(features))
	for i, v := range features {
		f32[i] = float32(v)
	}
	out, _, err := e.run(e.mlp, ort.NewShape(1, int64(len(f32))), f32)
	if err != nil {
		return nil, err
	}
	if len(out) != len(e.cfg.ClassNames) {
		return nil, fmt.Errorf("classifier mengeluarkan %d nilai, seharusnya %d kelas", len(out), len(e.cfg.ClassNames))
	}
	probs := make([]float64, len(out))
	for i, v := range out {
		probs[i] = float64(v)
	}
	return probs, nil
}