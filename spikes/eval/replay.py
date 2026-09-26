# -*- coding: utf-8 -*-
"""저장된 캡처(runs/<id>)를 폰 없이 서버 파이프라인에 다시 넣어 재실험 + 자동 채점.

사용: python3 replay.py --run ../a-live-ar/runs/20260926-120046 [--engine auto|gemini|openai] [--both] [--server http://localhost:8787]
meta.json 에 request_eyes / request_ref_eye 가 있어야 함 (2026-09-26 이후 캡처). 결과는 새 run 폴더에 저장되고 점수는 서버 로그·meta 에 남음.
"""
import argparse, base64, json, os, sys, time
import requests


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--engine", default="auto")
    ap.add_argument("--both", action="store_true"); ap.add_argument("--server", default="http://localhost:8787"); ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    m = json.load(open(os.path.join(a.run, "meta.json")))
    if not m.get("request_eyes"): sys.exit("이 run 에는 눈 기하가 저장돼 있지 않아요 (request_eyes 없음)")
    b64 = lambda name: base64.b64encode(open(os.path.join(a.run, name), "rb").read()).decode()
    body = {"frame": "data:image/jpeg;base64," + b64("frame.jpg"), "reference": "data:image/jpeg;base64," + b64("reference.jpg"),
            "engine": a.engine, "lashmap": m.get("lashmap"), "geom": m.get("geom"), "eyes": m["request_eyes"], "refEye": m["request_ref_eye"], "both": a.both}
    for i in range(a.n):
        t = time.time(); r = requests.post(a.server + "/api/tryon", json=body, timeout=600); res = r.json()
        if r.status_code != 200: print("ERR", res); continue
        j = res.get("judge") or {}
        print(f"[{i+1}/{a.n}] run {res['run_id']} · {time.time()-t:.0f}s · 유사도 {j.get('overall_similarity')} 질감 {j.get('texture_match')} 밀도 {j.get('density_match')} 실제감 {j.get('realism')} · {j.get('biggest_difference','')[:90]}")


if __name__ == "__main__":
    main()
