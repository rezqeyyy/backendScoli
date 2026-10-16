package main

import (
	"fmt"
	"math"
	"sort"
)

const rad2deg = 180.0 / math.Pi

// normalizeKeypoints sama dengan normalize_keypoints() di notebook:
// koordinat dikurangi rata-rata keypoint yang terlihat, lalu dibagi sebarannya.
func (c *Config) normalizeKeypoints(kpts [][3]float64) [][2]float64 {
	// Pusat dan skala hanya dari keypoint fitur (feature_kpts), sama dengan notebook.
	k := len(kpts)
	use := make([]bool, k)
	for _, i := range c.featureKpts() {
		use[i] = true
	}
	vis := make([]bool, k)
	n := 0
	for i := range kpts {
		if use[i] && kpts[i][2] > c.KptConf {
			vis[i] = true
			n++
		}
	}
	if n == 0 {
		copy(vis, use)
		for _, v := range use {
			if v {
				n++
			}
		}
	}
	var cx, cy float64
	for i := range kpts {
		if vis[i] {
			cx += kpts[i][0]
			cy += kpts[i][1]
		}
	}
	cx /= float64(n)
	cy /= float64(n)

	out := make([][2]float64, k)
	var ss float64
	for i := range kpts {
		if vis[i] { // titik tak terlihat tetap (0,0) = di pusat
			out[i] = [2]float64{kpts[i][0] - cx, kpts[i][1] - cy}
			ss += out[i][0]*out[i][0] + out[i][1]*out[i][1]
		}
	}
	scale := math.Max(math.Sqrt(ss/float64(n)), 1e-6)
	for i := range out {
		out[i][0] /= scale
		out[i][1] /= scale
	}
	return out
}

func (c *Config) axisPoints(p [][2]float64) [][2]float64 {
	a := make([][2]float64, 0, len(c.Schema.Axis))
	for _, n := range c.Schema.Axis {
		if n.Kind == "k" {
			a = append(a, p[n.Index])
		} else {
			l, r := c.Schema.Pairs[n.Index][0], c.Schema.Pairs[n.Index][1]
			a = append(a, [2]float64{(p[l][0] + p[r][0]) / 2, (p[l][1] + p[r][1]) / 2})
		}
	}
	// Urutkan dari atas ke bawah: model kadang menukar nomor ruas tulang belakang,
	// dan urutan yang tertukar membuat sudut antar-ruas palsu. Sama dengan axis_points() di notebook.
	sort.SliceStable(a, func(i, j int) bool { return a[i][1] < a[j][1] })
	return a
}

// PostureFeatures sama dengan posture_features() di notebook.
// kpts: [x, y, conf] tiap keypoint dalam piksel. Hasil: nama dan nilai fitur, urutannya selalu sama.
func (c *Config) PostureFeatures(kpts [][3]float64) ([]string, []float64) {
	p := c.normalizeKeypoints(kpts)
	var names []string
	var vals []float64
	add := func(n string, v float64) {
		names = append(names, n)
		vals = append(vals, v)
	}

	if c.useRawCoords() {
		for _, i := range c.featureKpts() {
			add("x_"+c.KptNames[i], p[i][0])
			add("y_"+c.KptNames[i], p[i][1])
		}
	}

	tilts := make([]float64, 0, len(c.Schema.Pairs))
	for i, pr := range c.Schema.Pairs {
		dx := p[pr[1]][0] - p[pr[0]][0]
		dy := p[pr[1]][1] - p[pr[0]][1]
		t := math.Atan2(dy, math.Abs(dx)+1e-9) * rad2deg
		tilts = append(tilts, t)
		add("tilt_"+c.PairNames[i]+"_deg", t)
		add("width_"+c.PairNames[i], math.Hypot(dx, dy))
	}
	if len(tilts) >= 2 {
		add("tilt_diff_top_bottom_deg", tilts[0]-tilts[len(tilts)-1])
	}

	a := c.axisPoints(p)
	n := len(a)
	var ax, ay, length float64
	if n >= 2 {
		ax, ay = a[n-1][0]-a[0][0], a[n-1][1]-a[0][1]
		length = math.Hypot(ax, ay) + 1e-9
		add("trunk_lean_deg", math.Atan2(ax, math.Abs(ay)+1e-9)*rad2deg)
	}
	if n >= 3 {
		minAng, maxAng := math.Inf(1), math.Inf(-1)
		for j := 0; j < n-1; j++ {
			sx, sy := a[j+1][0]-a[j][0], a[j+1][1]-a[j][1]
			ang := math.Atan2(sx, math.Abs(sy)+1e-9) * rad2deg
			add(fmt.Sprintf("seg_angle_%d_deg", j), ang)
			minAng, maxAng = math.Min(minAng, ang), math.Max(maxAng, ang)
		}
		minDev, maxDev, maxAbs := math.Inf(1), math.Inf(-1), 0.0
		for j := 1; j < n-1; j++ {
			vx, vy := a[j][0]-a[0][0], a[j][1]-a[0][1]
			dev := (ay*vx - ax*vy) / (length * length)
			add("lateral_dev_"+c.AxisNames[j], dev)
			minDev, maxDev = math.Min(minDev, dev), math.Max(maxDev, dev)
			maxAbs = math.Max(maxAbs, math.Abs(dev))
		}
		add("lateral_dev_max_abs", maxAbs)
		if n-2 >= 2 {
			add("lateral_dev_range", maxDev-minDev)
		}
		add("curve_angle_deg", maxAng-minAng)
	}
	return names, vals
}