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
}


def openai_gen(fb, rb, prompt, quality, fidelity="high"):
    key = os.environ["OPENAI_API_KEY"]
    files = [("image[]", ("frame.jpg", base64.b64decode(fb), "image/jpeg")), ("image[]", ("reference.jpg", base64.b64decode(rb), "image/jpeg"))]
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
    prompt = PROMPT_STYLES[v["prompt"]](meta.get("lashmap") or {}, meta.get("geom") or {"tipRatio": 0.45, "spikes": 7})
    k = key_of(v["model"], v["quality"], v["prompt"], v.get("fidelity", "high"), hashlib.sha1(fb.encode()).hexdigest()[:8], hashlib.sha1(rb.encode()).hexdigest()[:8], prompt)
    path = os.path.join(CACHE, k + ".png")
    if os.path.exists(path):
        return path, 0.0
    t = time.time()
    if v["model"] == "openai":
        b64 = openai_gen(fb, rb, prompt, v["quality"], v.get("fidelity", "high")); cost = PRICE["gen"]["openai"][v["quality"]]
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
    variants = [{"model": m, "quality": "low", "prompt": p} for m in models for p in PROMPT_STYLES]
    variants += [{"model": "openai", "quality": "low", "prompt": "sparse", "fidelity": "low"}]  # 얼굴 보존 낮추면 디자인 충실도 오르는지
    variants = variants[: a.explore]
    rows = []
    log("## 탐색 (quality low, mini 심사 1회)\n\n| 변형 | 유사도 | 길이 | 분포 | 질감 | 밀도 | 실제감 | 비용 |\n|---|---|---|---|---|---|---|---|")
    for v in variants:
        if spent >= a.budget: log(f"예산 소진 (${spent:.2f}) — 탐색 중단"); break
        try:
            path, c1 = generate(v, fb, rb, meta, log); spent += c1
            res, c2 = score(ref_img, path, "gpt-5.4-mini", 1); spent += c2
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
