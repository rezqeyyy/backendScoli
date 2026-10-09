package main

import (
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"log"

	"github.com/gofiber/fiber/v2"
	"golang.org/x/image/draw"
)

func main() {
	// 1. Setup GoFiber (Tanpa ONNX)
	app := fiber.New()

	// Endpoint untuk deteksi skoliosis
	app.Post("/api/v1/detect-scoliosis", handleDetection)

	fmt.Println("Server berjalan di port 3000 (Mode Tanpa ONNX)...")
	log.Fatal(app.Listen(":3000"))
}

func handleDetection(c *fiber.Ctx) error {
	// Ambil file gambar dari request
	file, err := c.FormFile("image")
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "Gambar tidak ditemukan di form-data"})
	}

	// Buka file gambar
	imgData, err := file.Open()
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "Gagal membuka gambar"})
	}
	defer imgData.Close()

	// Decode gambar (mendukung JPEG/PNG)
	img, _, err := image.Decode(imgData)
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "Format gambar tidak didukung"})
	}

	// Opsional: Lakukan pre-processing jika masih butuh manipulasi ukuran gambar
	_, err = preprocessImage(img)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "Gagal preprocessing gambar"})
	}

	// Contoh pemanggilan fungsi logika alternatif (misal dari predictor.go)
	// Anda bisa menyesuaikan input parameternya sesuai kebutuhan sistem Anda
	sampleInput := InputData{
		NilaiA: 75.0,
		NilaiB: 65.0,
		NilaiC: 80.0,
	}
	result := Predict(sampleInput)

	return c.JSON(fiber.Map{
		"status":  "success",
		"message": "Gambar berhasil diproses tanpa ONNX",
		"data": fiber.Map{
			"score":         result.Score,
			"diagnosis":     result.Status,
			"cobb_angle":    15.4, // Nilai dummy/simulasi sementara
		},
	})
}

// preprocessImage mengubah gambar menjadi flat array float32 jika masih diperlukan untuk hal lain
func preprocessImage(img image.Image) ([]float32, error) {
	bounds := img.Bounds()
	dst := image.NewRGBA(image.Rect(0, 0, 640, 640))
	
	// Resize menggunakan algoritma Bilinear
	draw.BiLinear.Scale(dst, dst.Bounds(), img, bounds, draw.Src, nil)

	tensor := make([]float32, 3*640*640)
	for y := 0; y < 640; y++ {
		for x := 0; x < 640; x++ {
			r, g, b, _ := dst.At(x, y).RGBA()
			
			tensor[(0*640*640)+(y*640)+x] = float32(r>>8) / 255.0
			tensor[(1*640*640)+(y*640)+x] = float32(g>>8) / 255.0
			tensor[(2*640*640)+(y*640)+x] = float32(b>>8) / 255.0
		}
	}

	return tensor, nil
}