"""눈 단위 합성(lashcomp) 실험: 배율 맞춘 눈 크롭 + 어두워진 픽셀만 합성. 이전 전체 프레임 방식과 비교.

사용: python3 eyecrop_test.py --run ../a-live-ar/runs/ref-fairy [--variants gemini-3-pro-image/sparse,...] [--both]
"""
import argparse
import base64
import hashlib
import io
import json
import os
import sys
import time

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "a-live-ar")); sys.path.insert(0, HERE)
import server  # noqa: E402
import judge  # noqa: E402
import lashcomp  # noqa: E402
import agent  # noqa: E402  (캐시된 _points, openai_gen, PRICE)
import requests  # noqa: E402

CACHE = agent.CACHE


def eye_points_full(fb, side):
    """프레임 절반 크롭에 gpt-5.5 점 찍기(캐시) → 프레임 픽셀 좌표 eye dict. side: 'left'|'right' (이미지 기준)."""
    img = Image.open(io.BytesIO(base64.b64decode(fb))).convert("RGB"); W, H = img.size
    x_off = 0 if side == "left" else W // 2
    half = img.crop((x_off, 0, x_off + W // 2, H)); buf = io.BytesIO(); half.save(buf, "JPEG", quality=92)
    m = agent._points(base64.b64encode(buf.getvalue()).decode(), f"frame-{side}-" + hashlib.sha1(fb.encode()).hexdigest()[:10])
    LW, LH = half.size
    P = lambda kx, ky: (m[kx] * LW + x_off, m[ky] * LH)
    return {"inner": P("inner_corner_x", "inner_corner_y"), "outer": P("outer_corner_x", "outer_corner_y"),
            "root": [P(f"root_x{j}", f"root_y{j}") for j in range(5)]}


def ref_framing(rb):
    """레퍼런스의 눈 너비/크롭 폭 비율과 눈 중심 위치 (0~1)."""
    img = Image.open(io.BytesIO(base64.b64decode(rb))); W, H = img.size
    m = agent._points(rb, hashlib.sha1(rb.encode()).hexdigest()[:10])
    inner = (m["inner_corner_x"] * W, m["inner_corner_y"] * H); outer = (m["outer_corner_x"] * W, m["outer_corner_y"] * H)
    w = ((outer[0] - inner[0]) ** 2 + (outer[1] - inner[1]) ** 2) ** 0.5
    return w / W, ((inner[0] + outer[0]) / 2 / W, (inner[1] + outer[1]) / 2 / H)


def crop_prompt(lm, geom, style):
    p = agent.PROMPT_STYLES[style](lm, geom)
    p = p.replace("Image 1 is a close-up photo of a person's eyes. Image 2 is a salon reference photo of eyelash extensions.",
                  "Image 1 is a close-up photo of ONE eye of a person. Image 2 is a salon reference photo of eyelash extensions on one eye, shown at the SAME SCALE as image 1 (both eyes are about the same width in the frame), so copy the lash lengths and thickness at the same pixel scale.")
    p = p.replace("Return only the edited version of image 1, same size and framing.",
                  "The eye in image 1 must stay at exactly the same position, size and shape; only add the lashes. Return only the edited version of image 1, same size and framing.")
    return p


def gen_one(model, quality, prompt, crop_b64, rb, mask_b64=None):
    if model == "openai":
        key = os.environ["OPENAI_API_KEY"]
        files = [("image[]", ("frame.jpg", base64.b64decode(crop_b64), "image/jpeg")), ("image[]", ("reference.jpg", base64.b64decode(rb), "image/jpeg"))]
        if mask_b64: files.append(("mask", ("mask.png", base64.b64decode(mask_b64), "image/png")))
        data = {"model": server.OPENAI_IMAGE_MODEL, "prompt": prompt + "\n(The first image is image 1, the second image is image 2.)",
                "size": "1536x1024", "quality": quality, "input_fidelity": "high", "n": "1"}
        r = requests.post("https://api.openai.com/v1/images/edits", headers={"Authorization": f"Bearer {key}"}, files=files, data=data, timeout=240)
        if r.status_code != 200: raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:200]}")
        return r.json()["data"][0]["b64_json"], agent.PRICE["gen"]["openai"][quality]
    server.GEMINI_IMAGE_MODEL = model
    return server.gemini_tryon(crop_b64, "image/jpeg", rb, "image/jpeg", prompt)[1], (0.13 if "pro" in model else 0.05)


def run_variant(v, frame, fb, rb, meta, ref_frac, ref_center, sides, log):
    lm, geom = meta.get("lashmap") or {}, meta.get("geom") or {}
    model, style = v.split("/")[0], v.split("/")[1]
    masked = v.endswith("/mask")
    prompt = crop_prompt(lm, geom, style)
    lower = bool(lm.get("lower_lashes"))
    result = frame.copy(); cost = 0.0; dbgs = {}
    for side in sides:
        eye = eye_points_full(fb, side)
        box = lashcomp.crop_box(eye, frame.width, frame.height, ref_frac, ref_center)
        cb = lashcomp.crop_b64(frame, box)
        k = agent.key_of("eyecrop", v, side, hashlib.sha1(cb.encode()).hexdigest()[:8], hashlib.sha1(rb.encode()).hexdigest()[:8], prompt)
        raw = os.path.join(CACHE, f"eyecrop-{k}-raw.png")
        if not os.path.exists(raw):
            t = time.time()
            mask = lashcomp.mask_png_b64(eye, box, lower) if masked else None
            b64, c = gen_one(model, "high", prompt, cb, rb, mask); cost += c
            open(raw, "wb").write(base64.b64decode(b64)); log(f"  gen {v} {side} {time.time()-t:.0f}s ${c:.2f}")
        out = Image.open(raw)
        result, dbg = lashcomp.composite_eye(result, eye, out, box, lower=lower)
        dbgs[side] = dbg
        dbg["crop_in"].save(os.path.join(CACHE, f"eyecrop-{k}-in.jpg"), quality=90)
        dbg["alpha"].save(os.path.join(CACHE, f"eyecrop-{k}-alpha.png"))
        log(f"  {side}: box {box} shift {dbg['shift']} alpha_mean {dbg['alpha_mean']:.4f}")
    path = os.path.join(CACHE, f"eyecrop-{agent.key_of(v, ''.join(sides), hashlib.sha1(fb.encode()).hexdigest()[:8])}.png")
    result.save(path); return path, cost, dbgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--variants", default="gemini-3-pro-image/sparse,gemini-3-pro-image/plain,gemini-3.1-flash-image/sparse,openai/sparse/mask")
    ap.add_argument("--both", action="store_true", help="양쪽 눈 (기본: 이미지 왼쪽 눈만 → 심사용)")
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    fb, rb, meta = agent.load_run(a.run)
    frame = Image.open(io.BytesIO(base64.b64decode(fb))).convert("RGB")
    ref_img = Image.open(io.BytesIO(base64.b64decode(rb))).convert("RGB")
    ref_frac, ref_center = ref_framing(rb)
    sides = ["left", "right"] if a.both else ["left"]
    ts = time.strftime("%Y%m%d-%H%M%S"); report = os.path.join(HERE, "reports", f"{ts}-eyecrop.md"); lines = []
    def log(s): print(s, flush=True); lines.append(s)
    log(f"# 눈 단위 합성 실험 {ts} · run {os.path.basename(a.run)} · 눈 {sides}\n")
    log(f"레퍼런스 프레이밍: 눈 너비/폭 {ref_frac:.2f}, 중심 ({ref_center[0]:.2f},{ref_center[1]:.2f})\n")
    log("| 변형 | 유사도 | 길이 | 분포 | 질감 | 밀도 | 아래 | 실제감 | 가장 큰 차이 |\n|---|---|---|---|---|---|---|---|---|")
    spent = 0.0; rows = []
    for v in a.variants.split(","):
        try:
            path, c, dbgs = run_variant(v, frame, fb, rb, meta, ref_frac, ref_center, sides, log); spent += c
            res, c2 = agent.score(ref_img, path, "gpt-5.5", a.n); spent += c2
            avg = res["avg"]; rows.append((avg["overall_similarity"], v, path, avg, res["runs"][0]["biggest_difference"], dbgs))
            log(f"| {v} | **{avg['overall_similarity']}** | {avg['length_match']} | {avg['shape_match']} | {avg['texture_match']} | {avg['density_match']} | {avg['lower_lash_match']} | {avg['realism']} | {res['runs'][0]['biggest_difference'][:100]} |")
        except Exception as e:  # noqa: BLE001
            log(f"| {v} | ERR {str(e)[:100]} | | | | | | | |")
    rows.sort(key=lambda r: -r[0])
    log(f"\n지출 약 ${spent:.2f}")
    if rows: log(f"최고: {rows[0][1]} → {rows[0][0]} · 파일 {rows[0][2]}")
    open(report, "w").write("\n".join(lines)); print("report:", report)
    json.dump([{"v": r[1], "path": r[2], "avg": r[3]} for r in rows], open(report.replace(".md", ".json"), "w"))


if __name__ == "__main__":
    main()
