"""전이 합성 실험: 캐시된 생성 결과(eyecrop-*-raw.png)에서 속눈썹만 뽑아 사용자 눈꺼풀에 워프해 붙이고 채점.

사용: python3 transfer_test.py --run ../a-live-ar/runs/ref-fairy
"""
import argparse
import base64
import glob
import hashlib
import io
import json
import os
import sys
import time

import requests
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "a-live-ar")); sys.path.insert(0, HERE)
import server  # noqa: E402
import agent  # noqa: E402
import lashcomp  # noqa: E402
import eyecrop_test as et  # noqa: E402

CACHE = agent.CACHE
EXT_SCHEMA = json.loads(json.dumps(server.EYEPOINTS_SCHEMA))
for j in range(5):
    EXT_SCHEMA["properties"][f"lower_x{j}"] = {"type": "number"}; EXT_SCHEMA["properties"][f"lower_y{j}"] = {"type": "number"}
    EXT_SCHEMA["required"] += [f"lower_x{j}", f"lower_y{j}"]
EXT_PROMPT = server.EYEPOINTS_PROMPT + "- lower_0..lower_4: 아랫눈꺼풀 라인(눈 구멍의 아래 경계, 아래 속눈썹 뿌리)을 눈앞(lower_0)에서 눈꼬리(lower_4)까지 등간격 5점.\n"


def points_ext(img_b64, cache_key):
    jp = os.path.join(CACHE, "ptsx-" + cache_key + ".json")
    if os.path.exists(jp): return json.load(open(jp))
    key = os.environ["OPENAI_API_KEY"]
    body = {"model": server.OPENAI_POINTS_MODEL, "reasoning_effort": server.OPENAI_POINTS_EFFORT,
            "messages": [{"role": "user", "content": [{"type": "text", "text": EXT_PROMPT}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "high"}}]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "eye_points_ext", "strict": True, "schema": EXT_SCHEMA}}}
    r = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=180)
    if r.status_code != 200: raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:300]}")
    m = json.loads(r.json()["choices"][0]["message"]["content"]); json.dump(m, open(jp, "w")); return m


def to_eye(m, W, H, x_off=0):
    P = lambda kx, ky: (m[kx] * W + x_off, m[ky] * H)
    return {"inner": P("inner_corner_x", "inner_corner_y"), "outer": P("outer_corner_x", "outer_corner_y"),
            "root": [P(f"root_x{j}", f"root_y{j}") for j in range(5)], "lower": [P(f"lower_x{j}", f"lower_y{j}") for j in range(5)]}


def jpeg_b64(img, q=92):
    buf = io.BytesIO(); img.convert("RGB").save(buf, "JPEG", quality=q); return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--raws", default=None, help="raw png 경로들(쉼표). 기본: 최근 eyecrop 실험의 4개")
    a = ap.parse_args()
    fb, rb, meta = agent.load_run(a.run)
    frame = Image.open(io.BytesIO(base64.b64decode(fb))).convert("RGB"); W, H = frame.size
    ref_img = Image.open(io.BytesIO(base64.b64decode(rb))).convert("RGB")
    lm = meta.get("lashmap") or {}; lower = bool(lm.get("lower_lashes"))
    ref_frac, ref_center = et.ref_framing(rb)
    # 사용자 왼쪽(이미지 기준) 눈: 절반 크롭에서 확장 점 (아랫눈꺼풀 포함)
    half = frame.crop((0, 0, W // 2, H))
    m = points_ext(jpeg_b64(half), "frame-left-" + hashlib.sha1(fb.encode()).hexdigest()[:10])
    eye = to_eye(m, W // 2, H)
    box = lashcomp.crop_box(eye, W, H, ref_frac, ref_center)
    raws = a.raws.split(",") if a.raws else sorted(glob.glob(os.path.join(CACHE, "eyecrop-*-raw.png")), key=os.path.getmtime)[-4:]
    names = json.load(open(sorted(glob.glob(os.path.join(HERE, "reports", "*-eyecrop.json")))[-1])) if not a.raws else []
    ts = time.strftime("%Y%m%d-%H%M%S"); report = os.path.join(HERE, "reports", f"{ts}-transfer.md"); lines = []
    def log(s): print(s, flush=True); lines.append(s)
    log(f"# 전이 합성 실험 {ts} · run {os.path.basename(a.run)} · box {box}\n")
    log("| 소스(생성 모델) | 유사도 | 길이 | 분포 | 질감 | 밀도 | 아래 | 실제감 | 가장 큰 차이 |\n|---|---|---|---|---|---|---|---|---|")
    spent = 0.0; rows = []
    for raw in raws:
        k = os.path.basename(raw).replace("-raw.png", "").replace("eyecrop-", "")
        out = Image.open(raw).convert("RGB").resize((lashcomp.OUT_W, lashcomp.OUT_H), Image.LANCZOS)
        try:
            mo = points_ext(jpeg_b64(out), "out-" + k); spent += 0.02
            eye_out = to_eye(mo, lashcomp.OUT_W, lashcomp.OUT_H)
            res, dbg = lashcomp.transfer_eye(frame, eye, out, eye_out, box, lower=lower)
            buf = io.BytesIO(); res.save(buf, "PNG"); h8 = hashlib.sha1(buf.getvalue()).hexdigest()[:8]
            path = os.path.join(CACHE, f"transfer-{k}-{h8}.png"); open(path, "wb").write(buf.getvalue())  # 내용 해시 → 합성 파라미터 바뀌면 재심사
            dbg["comp"].save(os.path.join(CACHE, f"transfer-{k}-comp.jpg"), quality=90); dbg["lash_map"].save(os.path.join(CACHE, f"transfer-{k}-map.png"))
            sc, c2 = agent.score(ref_img, path, "gpt-5.5", a.n); spent += c2
            avg = sc["avg"]; rows.append((avg["overall_similarity"], k, path, avg))
            log(f"| {k} (blur {dbg['blur']:.1f}) | **{avg['overall_similarity']}** | {avg['length_match']} | {avg['shape_match']} | {avg['texture_match']} | {avg['density_match']} | {avg['lower_lash_match']} | {avg['realism']} | {sc['runs'][0]['biggest_difference'][:100]} |")
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            log(f"| {k} | ERR {str(e)[:100]} | | | | | | | |")
    log(f"\n지출 약 ${spent:.2f}")
    open(report, "w").write("\n".join(lines)); print("report:", report)


if __name__ == "__main__":
    main()
