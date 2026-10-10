package main

import (
	"fmt"
	"image"
	"math"
	"sort"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	segIoU      = 0.7 // ambang NMS, sama dengan bawaan Ultralytics
	segMaxCands = 300 // kandidat terbanyak yang masuk NMS
	segMaxKeep  = 20  // orang terbanyak yang dihitung mask-nya
)

type segCand struct {
	idx            int
	score          float32
	x1, y1, x2, y2 float64
}

func (a segCand) iou(b segCand) float64 {
	iw := math.Min(a.x2, b.x2) - math.Max(a.x1, b.x1)
	ih := math.Min(a.y2, b.y2) - math.Max(a.y1, b.y1)
	if iw <= 0 || ih <= 0 {
		return 0
	}
	inter := iw * ih
	return inter / ((a.x2-a.x1)*(a.y2-a.y1) + (b.x2-b.x1)*(b.y2-b.y1) - inter)
}

// PersonMask menjalankan YOLOv8-seg dan mengembalikan mask (size*size, 1 = orang) milik
// orang dengan mask terluas, atau nil bila tidak ada orang. Sama dengan person_mask() di notebook.
func (e *Engine) PersonMask(tensor []float32) ([]uint8, error) {
	size := e.cfg.ImgSize
	outs, shapes, err := e.run(e.seg, ort.NewShape(1, 3, int64(size), int64(size)), tensor, 2)
	if err != nil {
		return nil, err
	}
	det, proto := outs[0], outs[1]
	if len(shapes[0]) != 3 || len(shapes[1]) != 4 || int(shapes[0][1]) != 4+e.segNC+e.segNM {
		return nil, fmt.Errorf("bentuk output segmentasi tidak terduga: %v %v", shapes[0], shapes[1])
	}
	n := int(shapes[0][2])
	mh, mw := int(shapes[1][2]), int(shapes[1][3])
	person := e.cfg.Background.PersonClass
	conf := float32(e.cfg.Background.SegConf)

	// Kandidat: anchor yang kelas terbaiknya "person" dan skornya cukup.
	var cands []segCand
	for i := 0; i < n; i++ {
		ps := det[(4+person)*n+i]
		if ps < conf {
			continue
		}
		isBest := true
		for c := 0; c < e.segNC; c++ {
			if c != person && det[(4+c)*n+i] > ps {
				isBest = false
				break
			}
		}
		if !isBest {
			continue
		}
		cx, cy, w, h := float64(det[i]), float64(det[n+i]), float64(det[2*n+i]), float64(det[3*n+i])
		cands = append(cands, segCand{i, ps, cx - w/2, cy - h/2, cx + w/2, cy + h/2})
	}
	if len(cands) == 0 {
		return nil, nil
	}
	sort.SliceStable(cands, func(a, b int) bool { return cands[a].score > cands[b].score })
	if len(cands) > segMaxCands {
		cands = cands[:segMaxCands]
	}

	// NMS: buang kotak yang menumpuk dengan kotak berskor lebih tinggi.
	var keep []segCand
	for _, c := range cands {
		ok := true
		for _, k := range keep {
			if c.iou(k) > segIoU {
				ok = false
				break
			}
		}
		if ok {
			keep = append(keep, c)
			if len(keep) == segMaxKeep {
				break
			}
		}
	}

	// Hitung mask tiap orang, ambil yang terluas.
	var best []uint8
	bestArea := 0
	logits := make([]float32, mh*mw)
	for _, c := range keep {
		for p := range logits {
			logits[p] = 0
		}
		for k := 0; k < e.segNM; k++ {
			coef := det[(4+e.segNC+k)*n+c.idx]
			plane := proto[k*mh*mw : (k+1)*mh*mw]
			for p, v := range plane {
				logits[p] += coef * v
			}
		}
		mask, area := upsampleMask(logits, mw, mh, size, c)
		if area > bestArea {
			best, bestArea = mask, area
		}
	}
	return best, nil
}

// upsampleMask memperbesar logit mask (mw x mh) ke size x size secara bilinear
// (pusat piksel sejajar, seperti F.interpolate align_corners=False), memotongnya ke kotak
// kandidat, lalu mengambil nilai > 0. Mengembalikan mask dan luasnya.
func upsampleMask(logits []float32, mw, mh, size int, c segCand) ([]uint8, int) {
	mask := make([]uint8, size*size)
	x0, x1 := clampInt(int(math.Ceil(c.x1)), 0, size), clampInt(int(math.Ceil(c.x2)), 0, size)
	y0, y1 := clampInt(int(math.Ceil(c.y1)), 0, size), clampInt(int(math.Ceil(c.y2)), 0, size)
	sx, sy := float64(mw)/float64(size), float64(mh)/float64(size)
	area := 0
	for y := y0; y < y1; y++ {
		fy := math.Max((float64(y)+0.5)*sy-0.5, 0)
		ya := int(fy)
		yb := min(ya+1, mh-1)
		wy := float32(fy - float64(ya))
		for x := x0; x < x1; x++ {
			fx := math.Max((float64(x)+0.5)*sx-0.5, 0)
			xa := int(fx)
			xb := min(xa+1, mw-1)
			wx := float32(fx - float64(xa))
			top := logits[ya*mw+xa]*(1-wx) + logits[ya*mw+xb]*wx
			bot := logits[yb*mw+xa]*(1-wx) + logits[yb*mw+xb]*wx
			if top*(1-wy)+bot*wy > 0 {
				mask[y*size+x] = 1
				area++
			}
		}
	}
	return mask, area
}

func clampInt(v, lo, hi int) int {
	return max(lo, min(v, hi))
}

// dilate melebarkan mask dengan jendela k x k (sama dengan cv2.dilate dengan kernel np.ones((k, k))).
func dilate(mask []uint8, w, h, k int) []uint8 {
	if k <= 1 {
		return mask
	}
	lo, hi := -(k / 2), k-1-k/2
	tmp := make([]uint8, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for d := lo; d <= hi; d++ {
				if xx := x + d; xx >= 0 && xx < w && mask[y*w+xx] != 0 {
					tmp[y*w+x] = 1
					break
				}
			}
		}
	}
	out := make([]uint8, w*h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			for d := lo; d <= hi; d++ {
				if yy := y + d; yy >= 0 && yy < h && tmp[yy*w+x] != 0 {
					out[y*w+x] = 1
					break
				}
			}
		}
	}
	return out
}

// RemoveBackground mengganti semua piksel di luar siluet orang dengan warna abu-abu polos.
// Sama dengan remove_background() di notebook. Mengembalikan false (gambar tidak diubah)
// bila fitur dimatikan, tidak ada orang, atau mask terlalu kecil.
func (e *Engine) RemoveBackground(img *image.RGBA) (bool, error) {
	bg := e.cfg.Background
	if !bg.Enabled || e.seg == nil {
		return false, nil
	}
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	if w != e.cfg.ImgSize || h != e.cfg.ImgSize {
		return false, fmt.Errorf("gambar %dx%d, seharusnya %dx%d", w, h, e.cfg.ImgSize, e.cfg.ImgSize)
	}
	mask, err := e.PersonMask(toTensor(img))
	if err != nil || mask == nil {
		return false, err
	}
	mask = dilate(mask, w, h, bg.MaskDilate)
	area := 0
	for _, v := range mask {
		area += int(v)
	}
	if float64(area)/float64(w*h) < bg.MinMaskFrac {
		return false, nil
	}
	c := uint8(bg.BgColor)
	for y := 0; y < h; y++ {
		row := img.Pix[y*img.Stride:]
		for x := 0; x < w; x++ {
			if mask[y*w+x] == 0 {
				row[4*x], row[4*x+1], row[4*x+2], row[4*x+3] = c, c, c, 255
			}
		}
	}
	return true, nil
}