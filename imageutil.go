package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	_ "image/jpeg"
	_ "image/png"
	"math"

	"golang.org/x/image/draw"
)

const maxPixels = 64_000_000 // 64 megapiksel

// decodeImage membaca JPEG/PNG menjadi RGBA dan menerapkan orientasi EXIF.
// Foto dari kamera ponsel sering disimpan "miring" dengan tanda orientasi di EXIF;
// tanpa langkah ini model akan menerima punggung yang rebah.
func decodeImage(data []byte) (*image.RGBA, error) {
	// Tolak dimensi yang tidak wajar sebelum mengalokasikan memori untuk seluruh piksel.
	if c, _, err := image.DecodeConfig(bytes.NewReader(data)); err != nil {
		return nil, err
	} else if c.Width*c.Height > maxPixels {
		return nil, fmt.Errorf("gambar %dx%d terlalu besar", c.Width, c.Height)
	}
	src, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	b := src.Bounds()
	rgba := image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
	draw.Draw(rgba, rgba.Bounds(), src, b.Min, draw.Src)
	return applyOrientation(rgba, exifOrientation(data)), nil
}

// exifOrientation mengembalikan nilai tag Orientation (1-8) dari JPEG, atau 1 bila tidak ada.
func exifOrientation(data []byte) int {
	if len(data) < 4 || data[0] != 0xFF || data[1] != 0xD8 {
		return 1
	}
	for i := 2; i+4 <= len(data); {
		if data[i] != 0xFF {
			return 1
		}
		marker := data[i+1]
		if marker == 0xDA || marker == 0xD9 { // mulai data gambar / akhir file
			return 1
		}
		size := int(binary.BigEndian.Uint16(data[i+2:]))
		if size < 2 || i+2+size > len(data) {
			return 1
		}
		seg := data[i+4 : i+2+size]
		if marker == 0xE1 && len(seg) >= 14 && string(seg[:6]) == "Exif\x00\x00" {
			return tiffOrientation(seg[6:])
		}
		i += 2 + size
	}
	return 1
}

func tiffOrientation(t []byte) int {
	var bo binary.ByteOrder
	switch string(t[:2]) {
	case "II":
		bo = binary.LittleEndian
	case "MM":
		bo = binary.BigEndian
	default:
		return 1
	}
	off := int(bo.Uint32(t[4:8]))
	if off+2 > len(t) {
		return 1
	}
	n := int(bo.Uint16(t[off:]))
	for j := 0; j < n; j++ {
		e := off + 2 + 12*j
		if e+12 > len(t) {
			return 1
		}
		if bo.Uint16(t[e:]) == 0x0112 {
			if v := int(bo.Uint16(t[e+8:])); v >= 1 && v <= 8 {
				return v
			}
			return 1
		}
	}
	return 1
}

func applyOrientation(src *image.RGBA, o int) *image.RGBA {
	if o <= 1 || o > 8 {
		return src
	}
	w, h := src.Bounds().Dx(), src.Bounds().Dy()
	dw, dh := w, h
	if o >= 5 { // orientasi 5-8 menukar lebar dan tinggi
		dw, dh = h, w
	}
	dst := image.NewRGBA(image.Rect(0, 0, dw, dh))
	for y := 0; y < dh; y++ {
		for x := 0; x < dw; x++ {
			var sx, sy int
			switch o {
			case 2:
				sx, sy = w-1-x, y
			case 3:
				sx, sy = w-1-x, h-1-y
			case 4:
				sx, sy = x, h-1-y
			case 5:
				sx, sy = y, x
			case 6:
				sx, sy = y, h-1-x
			case 7:
				sx, sy = w-1-y, h-1-x
			case 8:
				sx, sy = w-1-y, x
			}
			si, di := src.PixOffset(sx, sy), dst.PixOffset(x, y)
			copy(dst.Pix[di:di+4], src.Pix[si:si+4])
		}
	}
	return dst
}

// validateInput sama dengan is_valid_input() di notebook: menolak gambar yang jelas
// tidak bisa dianalisis sebelum masuk ke model.
func validateInput(img *image.RGBA, g InputGuard) (bool, string) {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	if min(w, h) < g.MinSide {
		return false, fmt.Sprintf("Resolusi terlalu kecil (%dx%d).", w, h)
	}

	gray := make([]float64, w*h)
	var seen [256]bool
	var satSum float64
	for y := 0; y < h; y++ {
		row := img.Pix[y*img.Stride : y*img.Stride+4*w]
		for x := 0; x < w; x++ {
			r, gg, b := float64(row[4*x]), float64(row[4*x+1]), float64(row[4*x+2])
			v := math.Round(0.299*r + 0.587*gg + 0.114*b)
			gray[y*w+x] = v
			seen[int(v)] = true
			if mx, mn := max(r, gg, b), min(r, gg, b); mx > 0 {
				satSum += math.Round(255 * (mx - mn) / mx)
			}
		}
	}
	if g.RequireGrayscale && satSum/float64(w*h) > g.MaxSaturation {
		return false, "Gambar berwarna; gambar yang diharapkan grayscale."
	}
	unique := 0
	for _, s := range seen {
		if s {
			unique++
		}
	}
	if unique < g.MinUniqueGray {
		return false, "Gambar terlalu blok/pixelated atau kontras sangat rendah."
	}
	if laplacianStd(gray, w, h) < g.MinEdgeStd {
		return false, "Gambar terlalu polos, tidak ada struktur yang bisa dianalisis."
	}
	return true, ""
}

// laplacianStd: simpangan baku hasil filter Laplacian 3x3 (tepi dicerminkan, seperti OpenCV).
func laplacianStd(gray []float64, w, h int) float64 {
	reflect := func(i, n int) int {
		if i < 0 {
			return -i
		}
		if i >= n {
			return 2*n - 2 - i
		}
		return i
	}
	var sum, sumSq float64
	for y := 0; y < h; y++ {
		yu, yd := reflect(y-1, h), reflect(y+1, h)
		for x := 0; x < w; x++ {
			xl, xr := reflect(x-1, w), reflect(x+1, w)
			l := gray[yu*w+x] + gray[yd*w+x] + gray[y*w+xl] + gray[y*w+xr] - 4*gray[y*w+x]
			sum += l
			sumSq += l * l
		}
	}
	n := float64(w * h)
	mean := sum / n
	return math.Sqrt(math.Max(sumSq/n-mean*mean, 0))
}

// preprocess: stretch ke size x size (sama dengan praproses dataset), lalu ubah ke
// tensor float32 berurutan CHW (R, G, B) dengan nilai 0-1.
func preprocess(img *image.RGBA, size int) []float32 {
	dst := image.NewRGBA(image.Rect(0, 0, size, size))
	draw.BiLinear.Scale(dst, dst.Bounds(), img, img.Bounds(), draw.Src, nil)
	plane := size * size
	tensor := make([]float32, 3*plane)
	for i := 0; i < plane; i++ {
		tensor[i] = float32(dst.Pix[4*i]) / 255
		tensor[plane+i] = float32(dst.Pix[4*i+1]) / 255
		tensor[2*plane+i] = float32(dst.Pix[4*i+2]) / 255
	}
	return tensor
}