package main

import (
	"encoding/json"
	"fmt"
	"os"
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

// Config adalah isi pipeline_config.json yang diekspor notebook (Bagian 14).
type Config struct {
	ClassNames      []string   `json:"class_names"`
	NormalClass     string     `json:"normal_class"`
	RiskThresholds  []float64  `json:"risk_thresholds"`
	RiskLevels      []string   `json:"risk_levels"`
	Disclaimer      string     `json:"disclaimer"`
	ImgSize         int        `json:"img_size"`
	StretchToSquare bool       `json:"stretch_to_square"`
	DetConf         float64    `json:"det_conf"`
	KptConf         float64    `json:"kpt_conf"`
	MinVisibleKpts  int        `json:"min_visible_kpts"`
	ReviewThreshold float64    `json:"review_threshold"`
	KptShape        []int      `json:"kpt_shape"`
	KptNames        []string   `json:"kpt_names"`
	Schema          Schema     `json:"schema"`
	PairNames       []string   `json:"pair_names"`
	AxisNames       []string   `json:"axis_names"`
	FeatureNames    []string   `json:"feature_names"`
	DisplayFeatures []string   `json:"display_features"`
	InputGuard      InputGuard `json:"input_guard"`
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
	if len(c.PairNames) != len(c.Schema.Pairs) || len(c.AxisNames) != len(c.Schema.Axis) {
		return fmt.Errorf("pair_names/axis_names tidak cocok dengan schema")
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