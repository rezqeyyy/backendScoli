package main

import (
	"math"
	"testing"
)

// Temperature scaling pada probabilitas harus sama dengan softmax(logit / T).
func TestCalibrateMatchesLogitScaling(t *testing.T) {
	logits := []float64{4.0, -1.5}
	softmax := func(z []float64, temp float64) []float64 {
		m := math.Max(z[0]/temp, z[1]/temp)
		a, b := math.Exp(z[0]/temp-m), math.Exp(z[1]/temp-m)
		return []float64{a / (a + b), b / (a + b)}
	}
	raw := softmax(logits, 1)
	for _, temp := range []float64{0, 1, 2, 3.5} {
		want := softmax(logits, temperature(temp))
		got := calibrate(raw, temp)
		for i := range got {
			if math.Abs(got[i]-want[i]) > 1e-9 {
				t.Fatalf("T=%.1f: got %v want %v", temp, got, want)
			}
		}
	}
	// T > 1 membuat probabilitas tidak terlalu ekstrem
	if p := calibrate([]float64{0.99, 0.01}, 3); p[0] > 0.9 || p[0] < 0.5 {
		t.Fatalf("kalibrasi tidak melunakkan: %v", p)
	}
}

func TestEnvOverrides(t *testing.T) {
	t.Setenv("DET_CONF", "0.25")
	t.Setenv("KPT_CONF", "0.3")
	t.Setenv("MIN_VISIBLE_KPTS", "4")
	t.Setenv("MLP_TEMPERATURE", "2.5")
	t.Setenv("RISK_THRESHOLDS", "0.3, 0.6")
	t.Setenv("DETECT_ON_ORIGINAL", "false")
	c := &Config{DetConf: 0.5, KptConf: 0.5, MinVisibleKpts: 7, RiskThresholds: []float64{0.35, 0.65}}
	if err := c.applyEnvOverrides(); err != nil {
		t.Fatal(err)
	}
	if c.DetConf != 0.25 || c.KptConf != 0.3 || c.MinVisibleKpts != 4 || c.Calibration.MLPTemperature != 2.5 ||
		c.RiskThresholds[0] != 0.3 || c.RiskThresholds[1] != 0.6 || c.detectOnOriginal() {
		t.Fatalf("override tidak terpasang: %+v", c)
	}
}