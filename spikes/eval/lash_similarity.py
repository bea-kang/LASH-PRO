"""레퍼런스 vs 결과 속눈썹 유사도.

양쪽 이미지에서 같은 방식으로:
  1) gpt-5.5로 눈앞·눈꼬리·뿌리 라인 5점 검출 (server.eyepoints)
  2) 뿌리 라인 위 띠에서 어두운 가닥 매팅 (index.html extractRefLashes 포팅)
  3) 뿌리 라인을 따라 t별 가닥 길이(눈 너비 단위) 프로파일 추출 (analyzeStrands 포팅, 단순화)
그리고 프로파일을 비교해 0~100 점수.

사용: python3 lash_similarity.py <reference.jpg> <result.png> [--crop-result]
결과 사진은 양쪽 눈이 있으니 --crop-result 로 한쪽 눈(이미지 왼쪽 눈)만 잘라 비교.
"""
import base64
import io
import json
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "a-live-ar"))
import server  # noqa: E402  (.env 로드 + eyepoints)


def b64(img):
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=92); return base64.b64encode(buf.getvalue()).decode()


def detect_points(img):
    """(inner, outer, root[5]) in pixel coords. 이미지가 눈 하나를 크게 담고 있어야 정확."""
    up = max(1, 1024 / img.width)
    big = img.resize((int(img.width * up), int(img.height * up)))
    m = server.eyepoints(b64(big), "image/jpeg")
    W, H = img.size
    P = lambda x, y: np.array([x * W, y * H])
    inner = P(m["inner_corner_x"], m["inner_corner_y"]); outer = P(m["outer_corner_x"], m["outer_corner_y"])
    root = [P(m[f"root_x{j}"], m[f"root_y{j}"]) for j in range(5)]
    return inner, outer, root, m.get("confidence")


def lid_at(root, inner, outer, t):
    """뿌리 폴리라인(눈앞→눈꼬리) 위 t(0~1), 바깥은 접선 외삽."""
    pts = np.array(root); w = np.linalg.norm(outer - inner) or 1
    if t < 0:
        d = pts[0] - pts[1]; d /= (np.linalg.norm(d) or 1); return pts[0] + d * (-t) * w
    if t > 1:
        d = pts[-1] - pts[-2]; d /= (np.linalg.norm(d) or 1); return pts[-1] + d * (t - 1) * w
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1); total = seg.sum() or 1; d = t * total
    for i, L in enumerate(seg):
        if d <= L or i == len(seg) - 1:
            u = min(1, d / L) if L else 0
            return pts[i] + (pts[i + 1] - pts[i]) * u
        d -= L
    return pts[-1]


def profile(img, inner, outer, root, step=1 / 60):
    """t별 가닥 길이(눈 너비 단위). 뿌리에서 위(법선 ±40°)로 가장 길게 이어지는 어두운 런."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    H, W = g.shape
    axis = outer - inner; w = np.linalg.norm(axis) or 1; axis_n = axis / w
    perp = np.array([-axis_n[1], axis_n[0]])
    mid = (inner + outer) / 2
    if np.dot(perp, root[2] - mid) < 0: perp = -perp  # 윗눈꺼풀 쪽 = up
    # 배경 밝기: 띠 안 픽셀의 상위 30%
    ys, xs = np.mgrid[0:H, 0:W]
    rel = np.stack([xs - inner[0], ys - inner[1]], -1)
    tt = (rel @ axis_n) / w
    band_vals = []
    for t in np.arange(-0.3, 1.4, 0.05):
        L = lid_at(root, inner, outer, t)
        for k in np.arange(0.05, 0.6, 0.05):
            p = L + perp * k * w; x, y = int(p[0]), int(p[1])
            if 0 <= x < W and 0 <= y < H: band_vals.append(g[y, x])
    bg = np.percentile(band_vals, 70) if band_vals else 200
    def alpha_at(p):
        x, y = int(round(p[0])), int(round(p[1]))
        if x < 1 or y < 1 or x >= W - 1 or y >= H - 1: return 0.0
        v = g[y-1:y+2, x-1:x+2].min()  # 3x3 최소 밝기 = 가는 가닥도 잡힘
        return max(0.0, min(1.0, (bg - v - 18) / 55))
    def solid_at(p):
        """굵은 어두운 덩어리(머리카락·눈썹·눈동자)인지: 7x7 평균 알파가 높으면 가닥이 아님"""
        x, y = int(round(p[0])), int(round(p[1]))
        if x < 3 or y < 3 or x >= W - 3 or y >= H - 3: return True
        return (bg - g[y-3:y+4, x-3:x+4]).mean() > 45
    def run_len(p0, d, max_len):
        gap = 0; last = 0
        for s in range(0, int(max_len)):
            p = p0 + d * s
            if s > w * 0.08 and solid_at(p): break   # 덩어리에 들어가면 중단 (뿌리 근처는 제외)
            v = alpha_at(p)
            if v > 0.25: gap = 0; last = s
            else:
                gap += 1
                if gap > w * 0.06: break
        return last
    out = []
    for t in np.arange(-0.3, 1.4, step):
        # 로컬 법선
        a = lid_at(root, inner, outer, t - 0.04); b = lid_at(root, inner, outer, t + 0.04)
        tan = b - a; tan /= (np.linalg.norm(tan) or 1); nrm = np.array([-tan[1], tan[0]])
        if np.dot(nrm, perp) < 0: nrm = -nrm
        rootp = lid_at(root, inner, outer, t) + nrm * w * 0.01
        best = 0
        for ang in np.arange(-0.8, 0.81, 0.05):
            c, s_ = math.cos(ang), math.sin(ang)
            d = np.array([nrm[0] * c - nrm[1] * s_, nrm[0] * s_ + nrm[1] * c])
            best = max(best, run_len(rootp, d, w * 0.55))
        out.append((round(float(t), 3), best / w))
    return out


def features(prof):
    inside = [(t, l) for t, l in prof if 0 <= t <= 1]
    lens = np.array([l for _, l in inside])
    tip = float(np.mean(np.sort(lens)[-max(1, len(lens) // 10):])) if len(lens) else 0
    spikes = 0; last = -1
    for j in range(1, len(inside) - 1):
        if inside[j][1] > inside[j-1][1] and inside[j][1] >= inside[j+1][1] and inside[j][1] > 0.22 and inside[j][0] - last > 0.06:
            spikes += 1; last = inside[j][0]
    # 구간별 평균 길이 (눈앞→눈꼬리 5구간)
    zones = []
    for z in range(5):
        seg = [l for t, l in inside if z / 5 <= t < (z + 1) / 5]
        zones.append(float(np.mean(seg)) if seg else 0)
    return {"tip": tip, "spikes": spikes, "zones": zones, "mean": float(lens.mean()) if len(lens) else 0}


def similarity(fr, fo):
    """0~100. 구간별 길이 차이(60%), 끝 높이 차이(25%), 스파이크 수 차이(15%)."""
    zd = np.mean([abs(a - b) for a, b in zip(fr["zones"], fo["zones"])])       # 눈 너비 단위, 0.1 = 10%
    td = abs(fr["tip"] - fo["tip"])
    sd = abs(fr["spikes"] - fo["spikes"])
    s = 100 * (0.6 * max(0, 1 - zd / 0.25) + 0.25 * max(0, 1 - td / 0.3) + 0.15 * max(0, 1 - sd / 6))
    return round(s, 1), {"zone_diff": round(float(zd), 3), "tip_diff": round(float(td), 3), "spike_diff": int(sd)}


def left_eye_crop(img, frac=0.5):
    """양쪽 눈이 있는 결과 사진에서 이미지 왼쪽 절반(한쪽 눈)만."""
    return img.crop((0, 0, int(img.width * frac), img.height))


if __name__ == "__main__":
    ref_path, out_path = sys.argv[1], sys.argv[2]
    ref = Image.open(ref_path).convert("RGB"); out = Image.open(out_path).convert("RGB")
    if "--crop-result" in sys.argv: out = left_eye_crop(out)
    ri, ro, rr, rc = detect_points(ref); oi, oo, orr, oc = detect_points(out)
    pr = profile(ref, ri, ro, rr); po = profile(out, oi, oo, orr)
    fr, fo = features(pr), features(po)
    score, detail = similarity(fr, fo)
    print(json.dumps({"score": score, "detail": detail, "reference": fr, "result": fo, "conf": [rc, oc]}, ensure_ascii=False, indent=1))
