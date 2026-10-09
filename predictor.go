package main

// InputData menampung parameter fitur dari client/frontend
type InputData struct {
	NilaiA float64 `json:"nilai_a"`
	NilaiB float64 `json:"nilai_b"`
	NilaiC float64 `json:"nilai_c"`
}

// ResultPrediction menampung hasil prediksi
type ResultPrediction struct {
	Score  float64 `json:"score"`
	Status string  `json:"status"`
}

// Predict menggunakan pendekatan bobot linear sederhana
func Predict(data InputData) ResultPrediction {
	bobotA := 0.4
	bobotB := 0.4
	bobotC := 0.2

	score := (data.NilaiA * bobotA) + (data.NilaiB * bobotB) + (data.NilaiC * bobotC)

	var status string
	if score >= 75 {
		status = "Rekomendasi Tinggi / Lolos"
	} else if score >= 50 {
		status = "Dipertimbangkan"
	} else {
		status = "Tidak Lolos"
	}

	return ResultPrediction{
		Score:  score,
		Status: status,
	}
}