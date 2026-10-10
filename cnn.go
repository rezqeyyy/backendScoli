package main

import (
	"image"
	"math"

	"golang.org/x/image/draw"
)

// cropForCNN sama dengan crop_for_cnn() di notebook: kotak punggung diperlebar `pad`,
// dipotong, lalu ditaruh di tengah kanvas abu-abu size x size tanpa mengubah rasio.
func cropForCNN(img *image.RGBA, box [4]float64, pad float64, size, bgColor int) *image.RGBA {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	bw, bh := box[2]-box[0], box[3]-box[1]
	x1 := int(math.Max(0, math.Floor(box[0]-pad*bw)))
	y1 := int(math.Max(0, math.Floor(box[1]-pad*bh)))
	x2 := int(math.Min(float64(w), math.Ceil(box[2]+pad*bw)))
	y2 := int(math.Min(float64(h), math.Ceil(box[3]+pad*bh)))
	if x2-x1 < 2 || y2-y1 < 2 { // kotak tidak wajar -> pakai seluruh gambar
		x1, y1, x2, y2 = 0, 0, w, h
	}
	cw, ch := x2-x1, y2-y1
	s := float64(size) / float64(max(cw, ch))
	nw := max(1, int(math.Floor(float64(cw)*s+0.5)))
	nh := max(1, int(math.Floor(float64(ch)*s+0.5)))

	out := image.NewRGBA(image.Rect(0, 0, size, size))
	c := uint8(bgColor)
	for i := 0; i < len(out.Pix); i += 4 {
		out.Pix[i], out.Pix[i+1], out.Pix[i+2], out.Pix[i+3] = c, c, c, 255
	}
	ox, oy := (size-nw)/2, (size-nh)/2
	src := image.Rect(x1, y1, x2, y2).Add(img.Bounds().Min)
	draw.BiLinear.Scale(out, image.Rect(ox, oy, ox+nw, oy+nh), img, src, draw.Src, nil)
	return out
}