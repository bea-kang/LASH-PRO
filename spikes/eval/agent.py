"""품질 실험 에이전트: 캡처 1장 + 레퍼런스 1장으로 변형을 탐색하고 채점해 최고 설정을 찾는다.

싸게 탐색(quality low + mini 심사 1회) → 상위만 비싸게 확정(quality high + 5.5 심사 3회).
같은 조합은 캐시로 재사용. 예산(USD) 안에서 멈춤. 결과는 reports/<ts>.md.

사용:
  python3 agent.py --run runs/20260922-191835 --budget 1.0 [--target 3.5] [--models openai] [--explore 8]
"""
import argparse
import base64
import hashlib
import io
import itertools
import json
import os
import sys
import time

import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "a-live-ar")); sys.path.insert(0, HERE)
import server  # noqa: E402
import judge  # noqa: E402

# 대략 단가 (USD). 실제 청구는 OpenAI 대시보드 기준. 2026-09 추정치.
PRICE = {
    "gen": {"openai": {"low": 0.03, "medium": 0.08, "high": 0.22}, "gemini": {"any": 0.05}},
    "judge": {"gpt-5.4-mini": 0.004, "gpt-5.5": 0.03},
}
CACHE = os.path.join(HERE, "cache")


def key_of(*parts):
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def load_run(run_dir):
    fb = base64.b64encode(open(os.path.join(run_dir, "frame.jpg"), "rb").read()).decode()
    rb = base64.b64encode(open(os.path.join(run_dir, "reference.jpg"), "rb").read()).decode()
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    return fb, rb, meta


# ---------- 보조 입력 생성 (레퍼런스 매트 / 프레임 틱 가이드) ----------
import numpy as np


def _points(img_b64_jpg, cache_key):
    """gpt-5.5로 눈앞·눈꼬리·뿌리 5점 (캐시)."""
    jp = os.path.join(CACHE, "pts-" + cache_key + ".json")
    if os.path.exists(jp):
        return json.load(open(jp))
    m = server.eyepoints(img_b64_jpg, "image/jpeg"); json.dump(m, open(jp, "w")); return m


def reference_matte(rb):
    """레퍼런스에서 속눈썹만 흰 배경 위에 (뿌리 라인 위 띠의 어두운 픽셀). PNG base64."""
    img = Image.open(io.BytesIO(base64.b64decode(rb))).convert("RGB"); W, H = img.size
    m = _points(rb, hashlib.sha1(rb.encode()).hexdigest()[:10])
    inner = np.array([m["inner_corner_x"] * W, m["inner_corner_y"] * H]); outer = np.array([m["outer_corner_x"] * W, m["outer_corner_y"] * H])
    root = [np.array([m[f"root_x{j}"] * W, m[f"root_y{j}"] * H]) for j in range(5)]
    g = np.asarray(img.convert("L"), dtype=np.float32)
    w = np.linalg.norm(outer - inner) or 1; axis = (outer - inner) / w; perp = np.array([-axis[1], axis[0]])
    if np.dot(perp, root[2] - (inner + outer) / 2) < 0: perp = -perp
    ys, xs = np.mgrid[0:H, 0:W]; P = np.stack([xs, ys], -1).astype(np.float32)
    # 뿌리 폴리라인까지의 법선 거리 근사: 각 x에서 root 폴리라인 y를 보간
    rx = np.array([p[0] for p in root]); ry = np.array([p[1] for p in root]); order = np.argsort(rx)
    lid_y = np.interp(xs[0], rx[order], ry[order])[None, :].repeat(H, 0)
    dv = (lid_y - ys) * (1 if perp[1] < 0 else -1)   # 양수 = 뿌리 라인에서 속눈썹 쪽
    t = ((xs - inner[0]) * axis[0] + (ys - inner[1]) * axis[1]) / w
    band = (dv > w * 0.02) & (dv < w * 0.75) & (t > -0.3) & (t < 1.4)
    bg = np.percentile(g[band], 70) if band.any() else 200
    alpha = np.clip((bg - g - 18) / 55, 0, 1) * band
    out = np.full((H, W, 3), 255, np.uint8); a = alpha[..., None]
    rgb = np.asarray(img, dtype=np.float32) * 0.85
    out = (rgb * a + 255 * (1 - a)).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(out).save(buf, "PNG"); return base64.b64encode(buf.getvalue()).decode(), alpha, (inner, outer, root, w, perp)


def spike_profile(alpha, geo):
    """매트 알파에서 t별 스파이크 길이(눈 너비 단위) → 스파이크 목록 [(t, len)]."""
    inner, outer, root, w, perp = geo; H, W = alpha.shape
    axis = (outer - inner) / w
    rx = np.array([p[0] for p in root]); ry = np.array([p[1] for p in root]); order = np.argsort(rx)
    prof = []
    for t in np.arange(0.0, 1.0, 1 / 40):
        x = inner[0] + axis[0] * t * w; y0 = np.interp(x, rx[order], ry[order])
        best = 0
        for ang in np.arange(-0.6, 0.61, 0.15):
            d = np.array([perp[0] * np.cos(ang) - perp[1] * np.sin(ang), perp[0] * np.sin(ang) + perp[1] * np.cos(ang)])
            last = 0; gap = 0
            for s_ in range(int(w * 0.7)):
                px, py = int(x + d[0] * s_), int(y0 + d[1] * s_)
                if px < 0 or py < 0 or px >= W or py >= H: break
                if alpha[py, px] > 0.3: last = s_; gap = 0
                else:
                    gap += 1
                    if gap > w * 0.05: break
            best = max(best, last)
        prof.append((float(t), best / w))
    # 지역 최대 = 스파이크
    spikes = [(t, l) for i, (t, l) in enumerate(prof) if 0 < i < len(prof) - 1 and l >= prof[i-1][1] and l > prof[i+1][1] and l > 0.18]
    return prof, spikes


def frame_with_ticks(fb, spikes):
    """프레임의 왼쪽 눈(이미지 기준)에 스파이크 위치·길이를 얇은 검은 선으로 그림 → JPEG base64."""
    img = Image.open(io.BytesIO(base64.b64decode(fb))).convert("RGB"); W, H = img.size
    left = img.crop((0, 0, W // 2, H)); buf = io.BytesIO(); left.save(buf, "JPEG", quality=92)
    m = _points(base64.b64encode(buf.getvalue()).decode(), "frame-left-" + hashlib.sha1(fb.encode()).hexdigest()[:10])
    LW, LH = left.size
    inner = np.array([m["inner_corner_x"] * LW, m["inner_corner_y"] * LH]); outer = np.array([m["outer_corner_x"] * LW, m["outer_corner_y"] * LH])
    root = [np.array([m[f"root_x{j}"] * LW, m[f"root_y{j}"] * LH]) for j in range(5)]
    w = np.linalg.norm(outer - inner) or 1; axis = (outer - inner) / w; perp = np.array([-axis[1], axis[0]])
    if np.dot(perp, root[2] - (inner + outer) / 2) < 0: perp = -perp
    d = ImageDraw.Draw(img)
    rx = np.array([p[0] for p in root]); ry = np.array([p[1] for p in root]); order = np.argsort(rx)
    for t, L in spikes:
        x = inner[0] + axis[0] * t * w; y = float(np.interp(x, rx[order], ry[order]))
        tip = (x + perp[0] * L * w, y + perp[1] * L * w)
        d.line([(x, y), tip], fill=(15, 10, 10), width=max(2, int(w / 60)))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=92); return base64.b64encode(buf.getvalue()).decode()


from PIL import ImageDraw  # noqa: E402

# ---------- 변형 공간 ----------
PROMPT_STYLES = {
    "sparse": lambda m, g: server.build_prompt(m, g, False),
    "bold": lambda m, g: server.build_prompt(dict(m, drama=5), g, False),
    "plain": lambda m, g: (
        "Image 1 is a close-up photo of a person's eyes. Image 2 is a salon reference photo of eyelash extensions.\n"
        "Apply the exact lash design of image 2 to the person in image 1: same lengths relative to the eye, same pattern of long and short lashes, same density, same curl and color. Copy image 2, do not restyle.\n"
        "Keep everything else in image 1 unchanged. Return only the edited image 1."),
    "fewer": lambda m, g: server.build_prompt(m, g, False).replace("Copy these from image 2 precisely:",
        "Draw at most 8 lashes per upper lid, each clearly separated, matching image 2's spike positions. Leave skin visible between them.\nCopy these from image 2 precisely:"),
    # 그림 힌트 변형들
    "ticks": lambda m, g: server.build_prompt(m, g, False).replace("Image 1 is a close-up photo of a person's eyes.",
        "Image 1 is a close-up photo of a person's eyes with thin dark GUIDE LINES drawn on the left eye's upper lash line: each line marks where a lash spike goes and how long it is. Replace every guide line with a realistic lash spike of that exact position and length, and mirror the same pattern on the right eye. Remove the guide lines; do not add spikes where there is no line."),
    "matte": lambda m, g: server.build_prompt(m, g, False).replace("Image 2 is a salon reference photo of eyelash extensions.",
        "Image 2 is a salon reference photo of eyelash extensions, and image 3 shows ONLY those lashes isolated on white — use image 3 as the exact pattern of spikes (count, spacing, relative lengths) to reproduce."),
    "twostage": None,  # generate()에서 특별 처리
}
STAGE2_PROMPT = """Image 1 is a photo of a person already wearing eyelash extensions. Image 2 is the salon reference design.
Edit ONLY the lashes in image 1 so their pattern matches image 2 exactly: turn the even fan into separated wispy SPIKES — the same number of spikes, at the same positions along the lid, with the same relative lengths; keep short sparse lashes between spikes and visible skin gaps. Do not make it fuller. Keep the face, eyes, skin, lighting and framing of image 1 unchanged. Return only the edited image 1."""


def openai_gen(fb, rb, prompt, quality, fidelity="high", extra=None):
    key = os.environ["OPENAI_API_KEY"]
    files = [("image[]", ("frame.jpg", base64.b64decode(fb), "image/jpeg")), ("image[]", ("reference.jpg", base64.b64decode(rb), "image/jpeg"))]
    if extra:  # 세 번째 이미지 (예: 속눈썹만 분리한 매트)
        files.append(("image[]", ("guide.png", base64.b64decode(extra), "image/png")))
    data = {"model": server.OPENAI_IMAGE_MODEL, "prompt": prompt + "\n(The first image is image 1, the second image is image 2.)",
            "size": "1536x1024", "quality": quality, "input_fidelity": fidelity, "n": "1"}
    r = requests.post("https://api.openai.com/v1/images/edits", headers={"Authorization": f"Bearer {key}"}, files=files, data=data, timeout=240)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:200]}")
    return r.json()["data"][0]["b64_json"]


def gemini_gen(fb, rb, prompt, model):
    server.GEMINI_IMAGE_MODEL = model
    return server.gemini_tryon(fb, "image/jpeg", rb, "image/jpeg", prompt)[1]


def generate(v, fb, rb, meta, log):
    """variant → (png_path, cost). 캐시 우선."""
    lm, geom = meta.get("lashmap") or {}, meta.get("geom") or {}
    pv = v["prompt"]
    prompt = STAGE2_PROMPT if pv == "twostage" else PROMPT_STYLES[pv](lm, geom)
    k = key_of(v["model"], v["quality"], pv, v.get("fidelity", "high"), hashlib.sha1(fb.encode()).hexdigest()[:8], hashlib.sha1(rb.encode()).hexdigest()[:8], prompt)
    path = os.path.join(CACHE, k + ".png")
    if os.path.exists(path):
        return path, 0.0
    t = time.time(); cost = 0.0
    if v["model"] == "openai":
        frame_in, extra = fb, None
        if pv in ("ticks", "matte"):
            matte_b64, alpha, geo = reference_matte(rb)
            if pv == "matte": extra = matte_b64
            else:
                _, spikes = spike_profile(alpha, geo); log(f"  spikes from reference: {len(spikes)} → {[(round(t_,2), round(l,2)) for t_, l in spikes]}")
                frame_in = frame_with_ticks(fb, spikes)
                open(os.path.join(CACHE, k + "-ticks-input.jpg"), "wb").write(base64.b64decode(frame_in))
        if pv == "twostage":
            base_v = dict(v, prompt="sparse"); base_path, c0 = generate(base_v, fb, rb, meta, log); cost += c0
            frame_in = base64.b64encode(open(base_path, "rb").read()).decode()
        b64 = openai_gen(frame_in, rb, prompt, v["quality"], v.get("fidelity", "high"), extra); cost += PRICE["gen"]["openai"][v["quality"]]
    else:
        b64 = gemini_gen(fb, rb, prompt, v["model"]); cost = PRICE["gen"]["gemini"]["any"]
    open(path, "wb").write(base64.b64decode(b64))
    log(f"  gen {v} {time.time()-t:.0f}s ${cost:.2f}")
    return path, cost


def score(ref_img, out_path, model, n):
    k = key_of("judge", model, n, os.path.basename(out_path))
    jp = os.path.join(CACHE, k + ".json")
    if os.path.exists(jp):
        return json.load(open(jp)), 0.0
    judge.JUDGE_MODEL = model
    out = judge.left_eye_crop(Image.open(out_path).convert("RGB"))
    res = judge.judge(ref_img, out, n=n)
    json.dump(res, open(jp, "w"), ensure_ascii=False)
    return res, PRICE["judge"][model] * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--target", type=float, default=3.5); ap.add_argument("--models", default="openai")
    ap.add_argument("--explore", type=int, default=8, help="탐색 변형 수 상한")
    ap.add_argument("--finalists", type=int, default=2)
    ap.add_argument("--prompts", default=None, help="탐색할 프롬프트 변형 (쉼표). 기본: 전부")
    a = ap.parse_args()
    fb, rb, meta = load_run(a.run)
    ref_img = Image.open(os.path.join(a.run, "reference.jpg")).convert("RGB")
    models = a.models.split(",")
    ts = time.strftime("%Y%m%d-%H%M%S"); report = os.path.join(HERE, "reports", f"{ts}.md"); lines = []
    def log(s): print(s, flush=True); lines.append(s)
    spent = 0.0
    log(f"# 실험 {ts} · run {os.path.basename(a.run)} · 예산 ${a.budget} · 목표 {a.target}\n")
    style = (meta.get("lashmap") or {}); log(f"레퍼런스: {style.get('style')} / {style.get('texture')} / drama {style.get('drama')}\n")
    # ---- 1) 탐색: low quality × 프롬프트 × 모델, mini 심사 1회 ----
    plist = a.prompts.split(",") if a.prompts else [p for p in PROMPT_STYLES]
    variants = [{"model": m, "quality": "low", "prompt": p} for m in models for p in plist]
    variants = variants[: a.explore]
    rows = []
    log("## 탐색 (quality low, 5.5 심사 1회)\n\n| 변형 | 유사도 | 길이 | 분포 | 질감 | 밀도 | 실제감 | 비용 |\n|---|---|---|---|---|---|---|---|")
    for v in variants:
        if spent >= a.budget: log(f"예산 소진 (${spent:.2f}) — 탐색 중단"); break
        try:
            path, c1 = generate(v, fb, rb, meta, log); spent += c1
            res, c2 = score(ref_img, path, "gpt-5.5", 1); spent += c2  # mini는 후하게 줘서 순위가 안 맞음 → 5.5 1회
            s = res["runs"][0]; rows.append((s["overall_similarity"], v, path, s))
            log(f"| {v['model']}/{v['prompt']}{'/fid-low' if v.get('fidelity')=='low' else ''} | {s['overall_similarity']} | {s['length_match']} | {s['shape_match']} | {s['texture_match']} | {s['density_match']} | {s['realism']} | ${c1+c2:.2f} |")
        except Exception as e:
            log(f"| {v['model']}/{v['prompt']} | ERR {str(e)[:80]} | | | | | | |")
    rows.sort(key=lambda r: -r[0])
    # ---- 2) 확정: 상위 N을 high로 재생성, 5.5 심사 3회 ----
    log(f"\n## 확정 (상위 {a.finalists} · quality high · 5.5 심사 3회)\n\n| 변형 | 유사도 | 길이 | 분포 | 질감 | 밀도 | 실제감 | 가장 큰 차이 |\n|---|---|---|---|---|---|---|---|")
    finals = []
    for sim, v, path, s in rows[: a.finalists]:
        if spent >= a.budget: log(f"예산 소진 (${spent:.2f}) — 확정 생략"); break
        try:
            hv = dict(v, quality="high"); hp, c1 = generate(hv, fb, rb, meta, log); spent += c1
            res, c2 = score(ref_img, hp, "gpt-5.5", 3); spent += c2
            avg = res["avg"]; finals.append((avg["overall_similarity"], hv, hp, avg, res["runs"][0]["biggest_difference"]))
            log(f"| {hv['model']}/{hv['prompt']}{'/fid-low' if hv.get('fidelity')=='low' else ''} | **{avg['overall_similarity']}** | {avg['length_match']} | {avg['shape_match']} | {avg['texture_match']} | {avg['density_match']} | {avg['realism']} | {res['runs'][0]['biggest_difference'][:90]} |")
        except Exception as e:
            log(f"| {v['model']}/{v['prompt']} | ERR {str(e)[:80]} | | | | | | |")
    finals.sort(key=lambda r: -r[0])
    log(f"\n## 결론\n")
    if finals:
        best = finals[0]
        verdict = "목표 달성" if best[0] >= a.target else "목표 미달"
        log(f"- 최고: {best[1]['model']}/{best[1]['prompt']} → 유사도 {best[0]} (질감 {best[3]['texture_match']}, 실제감 {best[3]['realism']}) — **{verdict}** (목표 {a.target})")
        log(f"- 최고 결과 파일: {best[2]}")
    log(f"- 지출 약 ${spent:.2f} (추정 단가 기준)")
    open(report, "w").write("\n".join(lines)); print("\nreport:", report)


if __name__ == "__main__":
    main()
