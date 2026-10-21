# -*- coding: utf-8 -*-
"""저장된 캡처(runs/<id>)를 폰 없이 서버 파이프라인에 다시 넣어 재실험 + 자동 채점.
여러 엔진/모델을 한 번에 돌려 표로 비교할 수 있다 (--engines).

사용:
  python3 replay.py --run ../a-live-ar/runs/20261004-160446
  python3 replay.py --run ../a-live-ar/runs/20261004-160446 --engines gemini:gemini-3-pro-image,gemini:gemini-3.1-flash-image,reference --n 3
엔진 표기: auto | reference | gemini[:모델] | openai[:모델]
meta.json 에 request_eyes / request_ref_eye 가 있어야 함 (2026-09-26 이후 캡처). 결과는 reports/<ts>-matrix.md.
"""
import argparse, base64, io, json, os, sys, time
import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "a-live-ar")); sys.path.insert(0, HERE)
import judge  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--engines", default="auto")
    ap.add_argument("--both", action="store_true"); ap.add_argument("--server", default="http://localhost:8787"); ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    m = json.load(open(os.path.join(a.run, "meta.json")))
    if not m.get("request_eyes"): sys.exit("이 run 에는 눈 기하가 저장돼 있지 않아요 (request_eyes 없음)")
    b64 = lambda name: base64.b64encode(open(os.path.join(a.run, name), "rb").read()).decode()
    ref_img = Image.open(os.path.join(a.run, "reference.jpg")).convert("RGB")
    base = {"frame": "data:image/jpeg;base64," + b64("frame.jpg"), "reference": "data:image/jpeg;base64," + b64("reference.jpg"),
            "lashmap": m.get("lashmap"), "geom": m.get("geom"), "eyes": m["request_eyes"], "refEye": m["request_ref_eye"], "both": a.both, "judge": False, "no_fallback": True}
    ts = time.strftime("%Y%m%d-%H%M%S"); lines = [f"# 모델 비교 {ts} · run {os.path.basename(a.run)} · 심사 {a.n}회 평균 (Claude)", "",
                                                   "| 엔진 | 유사도 | 길이 | 분포 | 질감 | 밀도 | 아래 | 실제감 | 초 | 결과 run | 가장 큰 차이 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    strips = []
    for spec in a.engines.split(","):
        engine, _, model = spec.partition(":")
        body = dict(base, engine=engine); body.update({"model": model} if model else {})
        t = time.time()
        try:
            r = requests.post(a.server + "/api/tryon", json=body, timeout=900); res = r.json()
            if r.status_code != 200: raise RuntimeError(res.get("error"))
            img = Image.open(io.BytesIO(base64.b64decode(res["image"].split(",", 1)[1]))).convert("RGB")
            side = next((k for k, v in res["eyes"].items() if "out" in v), next(iter(res["eyes"])))
            x0, y0, cw, ch = res["eyes"][side]["box"]
            crop = img.crop((x0, y0, x0 + cw, y0 + ch)).resize((1536, 1024))
            sc = judge.judge(ref_img, crop, n=a.n); avg = sc["avg"]
            lines.append(f"| {spec} | **{avg['overall_similarity']}** | {avg['length_match']} | {avg['shape_match']} | {avg['texture_match']} | {avg['density_match']} | {avg['lower_lash_match']} | {avg['realism']} | {time.time()-t:.0f} | {res['run_id']} | {sc['runs'][0]['biggest_difference'][:90]} |")
            strips.append((spec, crop.resize((768, 512))))
        except Exception as e:  # noqa: BLE001
            lines.append(f"| {spec} | ERR {str(e)[:120]} | | | | | | | {time.time()-t:.0f} | | |")
        print(lines[-1], flush=True)
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    rep = os.path.join(HERE, "reports", f"{ts}-matrix.md"); open(rep, "w").write("\n".join(lines) + "\n"); print("report:", rep)
    if strips:
        g = Image.new("RGB", (768 * (len(strips) + 1), 512), "white"); g.paste(ref_img.resize((768, 512)), (0, 0))
        for i, (_, im) in enumerate(strips): g.paste(im, (768 * (i + 1), 0))
        out = os.path.join(HERE, "reports", f"{ts}-matrix.jpg"); g.save(out, quality=88); print("strip:", out, "(순서: 레퍼런스,", ", ".join(s for s, _ in strips) + ")")


if __name__ == "__main__":
    main()
