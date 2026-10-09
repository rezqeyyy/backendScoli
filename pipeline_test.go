package main

// Uji kecocokan Go dengan Python. Jalankan setelah menyalin pipeline_config.json,
// test_vectors.json, dan kedua file .onnx dari folder ekspor notebook ke models/:
//
//	go test -v ./...
import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type testVector struct {
	File          string       `json:"file"`
	Keypoints     [][3]float64 `json:"keypoints_xyc"`
	Features      []float64    `json:"features"`
	Probabilities []float64    `json:"probabilities"`
}

func loadFixtures(t *testing.T) (*Config, []testVector, string) {
	dir := envOr("MODELS_DIR", "models")
	cfgPath := filepath.Join(dir, "pipeline_config.json")
	if _, err := os.Stat(cfgPath); err != nil {
		t.Skipf("pipeline_config.json belum ada di %s", dir)
	}
	c, err := LoadConfig(cfgPath)
	if err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(filepath.Join(dir, "test_vectors.json"))
	if err != nil {
		t.Skipf("test_vectors.json belum ada di %s", dir)
	}
	var vecs []testVector
	if err := json.Unmarshal(b, &vecs); err != nil {
		t.Fatalf("test_vectors.json tidak valid: %v", err)
	}
	return c, vecs, dir
}

// Fitur postur hasil Go harus sama dengan hasil posture_features() di notebook.
func TestFeaturesMatchPython(t *testing.T) {
	c, vecs, _ := loadFixtures(t)
	for _, v := range vecs {
		_, got := c.PostureFeatures(v.Keypoints)
		if len(got) != len(v.Features) {
			t.Fatalf("%s: %d fitur, Python %d", v.File, len(got), len(v.Features))
		}
		for i := range got {
			// keypoint di test_vectors dibulatkan 4 desimal, jadi toleransinya longgar sedikit
			if d := math.Abs(got[i] - v.Features[i]); d > 1e-2*math.Max(1, math.Abs(v.Features[i])) {
				t.Errorf("%s: fitur %s = %.6f, Python %.6f", v.File, c.FeatureNames[i], got[i], v.Features[i])
			}
		}
	}
	t.Logf("%d vektor uji, %d fitur: cocok", len(vecs), len(c.FeatureNames))
}

// Probabilitas dari scoliosis_mlp.onnx lewat Go harus sama dengan PyTorch di notebook.
func TestClassifierMatchesPython(t *testing.T) {
	c, vecs, dir := loadFixtures(t)
	e, err := NewEngine(c, envOr("ONNXRUNTIME_LIB", defaultLibPath()), dir)
	if err != nil {
		t.Skipf("model belum bisa dimuat: %v", err)
	}
	defer e.Close()
	for _, v := range vecs {
		got, err := e.Classify(v.Features)
		if err != nil {
			t.Fatal(err)
		}
		for i := range got {
			if math.Abs(got[i]-v.Probabilities[i]) > 1e-3 {
				t.Errorf("%s: P(%s) = %.5f, Python %.5f", v.File, c.ClassNames[i], got[i], v.Probabilities[i])
			}
		}
	}
	t.Logf("%d vektor uji: probabilitas cocok", len(vecs))
}