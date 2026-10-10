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

// Engine memegang sesi ONNX Runtime: YOLOv8 Pose dan MLP selalu ada;
// YOLOv8-seg (hapus background) dan CNN (ResNet) hanya dimuat bila diminta config.
type Engine struct {
	cfg      *Config
	pose     *ort.DynamicAdvancedSession
	mlp      *ort.DynamicAdvancedSession
	seg      *ort.DynamicAdvancedSession
	cnn      *ort.DynamicAdvancedSession
	poseRows int // 4 + jumlah kelas + keypoint x dimensi
	segNC    int // jumlah kelas model segmentasi (80 untuk COCO)
	segNM    int // jumlah koefisien mask (32)
	mu       sync.Mutex
}

// Detection adalah kandidat terbaik dari YOLOv8 Pose (koordinat piksel pada gambar img_size x img_size).
type Detection struct {
	Kpts  [][3]float64 // x, y, conf
	Box   [4]float64   // x1, y1, x2, y2
	Class int
	Conf  float64
}

// openSession membaca nama input/output model lalu membuka sesinya.
func openSession(path string, nIn, nOut int) (*ort.DynamicAdvancedSession, []ort.InputOutputInfo, []ort.InputOutputInfo, error) {
	in, out, err := ort.GetInputOutputInfo(path)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("gagal membaca %s: %w", path, err)
	}
	if len(in) != nIn || len(out) < nOut {
		return nil, nil, nil, fmt.Errorf("%s: %d input dan %d output, seharusnya %d input dan minimal %d output", path, len(in), len(out), nIn, nOut)
	}
	inNames, outNames := make([]string, nIn), make([]string, nOut)
	for i := range inNames {
		inNames[i] = in[i].Name
	}
	for i := range outNames {
		outNames[i] = out[i].Name
	}
	s, err := ort.NewDynamicAdvancedSession(path, inNames, outNames, nil)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("gagal memuat %s: %w", path, err)
	}
	return s, in, out, nil
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
	var in, out []ort.InputOutputInfo

	// Bentuk input/output dicek saat start, supaya model yang tidak cocok dengan config langsung ketahuan.
	posePath := filepath.Join(modelsDir, "scoliosis_pose.onnx")
	if e.pose, _, out, err = openSession(posePath, 1, 1); err != nil {
		return nil, err
	}
	if d := out[0].Dimensions; len(d) != 3 || (d[1] > 0 && int(d[1]) != e.poseRows) {
		return nil, fmt.Errorf("%s: bentuk output %v, seharusnya [1 %d N] sesuai kelas dan keypoint di config", posePath, d, e.poseRows)
	}

	mlpPath := filepath.Join(modelsDir, "scoliosis_mlp.onnx")
	if e.mlp, in, _, err = openSession(mlpPath, 1, 1); err != nil {
		return nil, err
	}
	if d := in[0].Dimensions; len(d) != 2 || int(d[1]) != len(cfg.FeatureNames) {
		return nil, fmt.Errorf("%s: bentuk input %v, seharusnya [N %d] sesuai feature_names", mlpPath, d, len(cfg.FeatureNames))
	}

	if cfg.Background.Enabled {
		segPath := filepath.Join(modelsDir, "scoliosis_seg.onnx")
		if e.seg, _, out, err = openSession(segPath, 1, 2); err != nil {
			return nil, fmt.Errorf("config meminta hapus background: %w", err)
		}
		d0, d1 := out[0].Dimensions, out[1].Dimensions
		if len(d0) != 3 || len(d1) != 4 {
			return nil, fmt.Errorf("%s: bentuk output %v dan %v bukan model YOLOv8-seg", segPath, d0, d1)
		}
		e.segNM = int(d1[1])
		e.segNC = int(d0[1]) - 4 - e.segNM
		if e.segNM <= 0 || cfg.Background.PersonClass >= e.segNC {
			return nil, fmt.Errorf("%s: %d kelas dan %d koefisien mask tidak cocok dengan person_class %d", segPath, e.segNC, e.segNM, cfg.Background.PersonClass)
		}
	}

	if cfg.UsesCNN() {
		cnnPath := filepath.Join(modelsDir, "scoliosis_cnn.onnx")
		if e.cnn, in, _, err = openSession(cnnPath, 1, 1); err != nil {
			return nil, fmt.Errorf("classifier akhir %q membutuhkan CNN: %w", cfg.FinalClassifier, err)
		}
		if d := in[0].Dimensions; len(d) != 4 || int(d[1]) != 3 || int(d[2]) != cfg.CNN.ImgSize || int(d[3]) != cfg.CNN.ImgSize {
			return nil, fmt.Errorf("%s: bentuk input %v, seharusnya [N 3 %d %d]", cnnPath, d, cfg.CNN.ImgSize, cfg.CNN.ImgSize)
		}
	}
	return e, nil
}

func (e *Engine) Close() {
	for _, s := range []*ort.DynamicAdvancedSession{e.pose, e.mlp, e.seg, e.cnn} {
		if s != nil {
			s.Destroy()
		}
	}
	ort.DestroyEnvironment()
}

// run menjalankan satu sesi dengan satu input float32 dan mengembalikan data serta bentuk
// tiap output (nOut harus sama dengan jumlah output saat sesi dibuka).
func (e *Engine) run(s *ort.DynamicAdvancedSession, shape ort.Shape, data []float32, nOut int) ([][]float32, []ort.Shape, error) {
	in, err := ort.NewTensor(shape, data)
	if err != nil {
		return nil, nil, err
	}
	defer in.Destroy()

	outs := make([]ort.Value, nOut)
	e.mu.Lock()
	err = s.Run([]ort.Value{in}, outs)
	e.mu.Unlock()
	defer func() {
		for _, o := range outs {
			if o != nil {
				o.Destroy()
			}
		}
	}()
	if err != nil {
		return nil, nil, err
	}

	res, shapes := make([][]float32, nOut), make([]ort.Shape, nOut)
	for i, o := range outs {
		t, ok := o.(*ort.Tensor[float32])
		if !ok {
			return nil, nil, fmt.Errorf("output model ke-%d bukan tensor float32", i)
		}
		res[i] = make([]float32, len(t.GetData()))
		copy(res[i], t.GetData()) // salin sebelum tensor dibebaskan
		shapes[i] = t.GetShape()
	}
	return res, shapes, nil
}

// Detect menjalankan YOLOv8 Pose dan mengambil kandidat dengan skor kelas tertinggi.
// Kandidat teratas selalu lolos NMS, jadi NMS tidak perlu dijalankan untuk mengambil satu deteksi terbaik.
// Mengembalikan nil bila skor terbaik di bawah det_conf.
func (e *Engine) Detect(tensor []float32) (*Detection, error) {
	s := int64(e.cfg.ImgSize)
	outs, shapes, err := e.run(e.pose, ort.NewShape(1, 3, s, s), tensor, 1)
	if err != nil {
		return nil, err
	}
	out, shape := outs[0], shapes[0]
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

// classify menjalankan satu model classifier dan mengembalikan probabilitas tiap kelas.
func (e *Engine) classify(s *ort.DynamicAdvancedSession, shape ort.Shape, data []float32) ([]float64, error) {
	outs, _, err := e.run(s, shape, data, 1)
	if err != nil {
		return nil, err
	}
	if len(outs[0]) != len(e.cfg.ClassNames) {
		return nil, fmt.Errorf("classifier mengeluarkan %d nilai, seharusnya %d kelas", len(outs[0]), len(e.cfg.ClassNames))
	}
	probs := make([]float64, len(outs[0]))
	for i, v := range outs[0] {
		probs[i] = float64(v)
	}
	return probs, nil
}

// Classify mengirim vektor fitur postur ke MLP.
func (e *Engine) Classify(features []float64) ([]float64, error) {
	f32 := make([]float32, len(features))
	for i, v := range features {
		f32[i] = float32(v)
	}
	return e.classify(e.mlp, ort.NewShape(1, int64(len(f32))), f32)
}

// ClassifyImage mengirim potongan gambar (tensor CHW, RGB, 0-1) ke CNN.
func (e *Engine) ClassifyImage(tensor []float32) ([]float64, error) {
	s := int64(e.cfg.CNN.ImgSize)
	return e.classify(e.cnn, ort.NewShape(1, 3, s, s), tensor)
}