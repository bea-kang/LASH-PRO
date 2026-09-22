"""spike dev server: static files + /api/lashmap (사진 -> 래시 맵 JSON) + /api/tryon (사진 그대로 합성).

Run:  python3 server.py  (port 8787)
Keys: 같은 폴더의 .env (gitignore됨). OPENAI_API_KEY 하나로 다 됨. GEMINI_API_KEY 는 비교용 선택.
"""
import base64
import json
import os
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import urllib.error
import urllib.request

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OPENAI_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4.1")
GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")
OPENAI_IMAGE_MODEL = os.environ.get("OPENAI_IMAGE_MODEL", "gpt-image-1")


def load_dotenv():
    """spikes/a-live-ar/.env 의 KEY=VALUE 를 환경변수로 (이미 있으면 건드리지 않음)."""
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


load_dotenv()

LASHMAP_SCHEMA = {
    "type": "object",
    "properties": {
        "is_lash_photo": {"type": "boolean"},
        "curl": {"type": "string", "enum": ["J", "C", "D"]},
        # structured output은 minItems>1, minimum/maximum을 안 받음 → 5구간 개별 필드 + 서버에서 클램프
        "len_inner_mm": {"type": "integer"},
        "len_inner_mid_mm": {"type": "integer"},
        "len_center_mm": {"type": "integer"},
        "len_outer_mid_mm": {"type": "integer"},
        "len_outer_mm": {"type": "integer"},
        "volume": {"type": "string", "enum": ["natural", "volume", "mega"]},
        "style": {"type": "string", "enum": ["natural", "cat", "doll"]},
        "texture": {"type": "string", "enum": ["even", "wispy"]},
        # 레퍼런스 눈 기준점 (이미지 폭·높이 대비 0~1). 눈이 여러 개면 가장 크고 선명한 눈 하나
        "inner_corner_x": {"type": "number"}, "inner_corner_y": {"type": "number"},
        "outer_corner_x": {"type": "number"}, "outer_corner_y": {"type": "number"},
        "lash_root_center_x": {"type": "number"}, "lash_root_center_y": {"type": "number"},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["is_lash_photo", "curl", "len_inner_mm", "len_inner_mid_mm", "len_center_mm", "len_outer_mid_mm", "len_outer_mm", "volume", "style", "texture", "inner_corner_x", "inner_corner_y", "outer_corner_x", "outer_corner_y", "lash_root_center_x", "lash_root_center_y", "confidence", "notes"],
    "additionalProperties": False,
}

PROMPT = """이 사진은 속눈썹 연장 샵의 시술 결과 레퍼런스입니다. 이 디자인을 다른 사람 눈에 재현하기 위한 래시 맵(시술 사양)으로 번역하세요.

- curl: J(거의 직선, 살짝 올라감) / C(표준 컬) / D(강한 컬, 눈꺼풀 쪽으로 많이 감김)
- len_inner_mm → len_inner_mid_mm → len_center_mm → len_outer_mid_mm → len_outer_mm: 눈앞(내안각)부터 눈꼬리(외안각)까지 5구간의 연장 길이(mm). 실제 샵 사양 범위(6~15mm)로. 사진의 자연 속눈썹 길이와 비교해 추정.
- volume: natural(1:1 클래식, 성근 느낌) / volume(2~4D, 촘촘) / mega(5D 이상, 매우 빽빽)
- style: natural(균일) / cat(눈꼬리로 갈수록 길어짐) / doll(중앙이 가장 김)
- texture: even(길이가 고르게 정렬된 클래식/볼륨) / wispy(몇 가닥씩 뾰족하게 모여 스파이크를 이루는 요정·위스피 스타일. 긴 스파이크와 짧은 베이스가 번갈아 보임)
- inner_corner / outer_corner / lash_root_center: 사진에서 가장 크고 선명한 눈 하나의 기준점. 이미지 폭·높이 대비 0~1 좌표. inner_corner는 코 쪽 눈구석(눈물점), outer_corner는 바깥 눈꼬리 끝, lash_root_center는 윗눈꺼풀 속눈썹 뿌리 라인의 한가운데(눈동자 위쪽 라인). 눈이 감겨 있어도 라인 기준으로.
- confidence: 0~1. 사진이 눈을 감고 있거나 접사라 열린 눈 상태를 추정해야 하면 낮게.
- notes: 한 줄. 특징(예: "눈꼬리 3구간만 길게, 컬러 브라운") 또는 추정 근거.
- 속눈썹 사진이 아니면 is_lash_photo=false 로 하고 나머지는 기본값(C, 8-9-10-10-9, natural, natural, even, 0).
"""


def translate(image_b64: str, media_type: str) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 없어요. spikes/a-live-ar/.env 에 OPENAI_API_KEY=sk-... 를 넣고 서버를 다시 띄우세요.")
    body = {
        "model": OPENAI_VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}", "detail": "high"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        "response_format": {"type": "json_schema", "json_schema": {"name": "lash_map", "strict": True, "schema": LASHMAP_SCHEMA}},
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:600]}")
    msg = r.json()["choices"][0]["message"]
    if msg.get("refusal"):
        raise RuntimeError("model refused: " + msg["refusal"])
    data = json.loads(msg["content"])
    data["zone_lengths_mm"] = [
        max(6, min(15, int(data.pop(k))))
        for k in ("len_inner_mm", "len_inner_mid_mm", "len_center_mm", "len_outer_mid_mm", "len_outer_mm")
    ]
    data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
    u = r.json().get("usage", {})
    data["_usage"] = {"in": u.get("prompt_tokens"), "out": u.get("completion_tokens")}
    return data


EYEPOINTS_SCHEMA = {
    "type": "object",
    "properties": {
        "inner_corner_x": {"type": "number"}, "inner_corner_y": {"type": "number"},
        "outer_corner_x": {"type": "number"}, "outer_corner_y": {"type": "number"},
        # 윗눈꺼풀 속눈썹 뿌리 라인: 눈앞→눈꼬리 등간격 5점
        "root_x0": {"type": "number"}, "root_y0": {"type": "number"},
        "root_x1": {"type": "number"}, "root_y1": {"type": "number"},
        "root_x2": {"type": "number"}, "root_y2": {"type": "number"},
        "root_x3": {"type": "number"}, "root_y3": {"type": "number"},
        "root_x4": {"type": "number"}, "root_y4": {"type": "number"},
        "confidence": {"type": "number"},
    },
    "required": ["inner_corner_x", "inner_corner_y", "outer_corner_x", "outer_corner_y",
                 "root_x0", "root_y0", "root_x1", "root_y1", "root_x2", "root_y2", "root_x3", "root_y3", "root_x4", "root_y4", "confidence"],
    "additionalProperties": False,
}

EYEPOINTS_PROMPT = """이 이미지는 한쪽 눈을 크게 자른 사진입니다. 이미지 폭·높이 대비 0~1 좌표로 다음을 정확히 찍으세요.
- inner_corner: 코 쪽 눈구석(눈물점). 눈 구멍(흰자/눈동자가 보이는 영역)의 안쪽 끝점.
- outer_corner: 바깥 눈꼬리. 눈 구멍의 바깥 끝점. 속눈썹 끝이 아니라 눈꺼풀이 만나는 지점.
- root_0..root_4: 윗눈꺼풀의 속눈썹이 피부에서 나오는 뿌리 라인을 눈앞(root_0)에서 눈꼬리(root_4)까지 등간격 5점. 이 라인은 눈 구멍의 위쪽 경계와 거의 같고, 속눈썹 가닥의 아래 끝입니다. 쌍꺼풀 선이나 눈썹이 아닙니다. 눈동자 바로 위를 지납니다.
- confidence: 0~1.
"""


def eyepoints(image_b64: str, media_type: str) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 없어요.")
    body = {
        "model": OPENAI_VISION_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}", "detail": "high"}},
            {"type": "text", "text": EYEPOINTS_PROMPT},
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "eye_points", "strict": True, "schema": EYEPOINTS_SCHEMA}},
    }
    r = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=120)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:600]}")
    msg = r.json()["choices"][0]["message"]
    if msg.get("refusal"):
        raise RuntimeError("model refused: " + msg["refusal"])
    return json.loads(msg["content"])


TRYON_PROMPT = """Image 1 is a close-up photo of a person's eyes. Image 2 is a reference photo of eyelash extensions done at a lash salon.
Edit image 1 so the person is wearing eyelash extensions that COPY image 2's lashes: the same curl, the same lengths per zone (inner corner to outer corner), the same density, the same spiky/wispy or even pattern, the same lash color.
Do not reinterpret, stylize, or "improve" the design. Do not make the lashes longer, thicker, or more dramatic than image 2. If image 2 has a few spikes, make the same number of spikes in the same places along the lid.
Keep everything else in image 1 exactly as it is: eye shape and size, eyelids, skin texture, iris, eyebrows, lighting, framing, resolution. Do not enlarge the eyes, do not retouch the skin, do not add eyeliner or other makeup, do not change the face.
Return only the edited version of image 1, same size and framing."""

CURL_DESC = {"J": "J curl (nearly straight, slight lift)", "C": "C curl (standard lift)", "D": "D curl (strong lift, curls up toward the lid)"}
VOL_DESC = {"natural": "classic 1:1, sparse", "volume": "2D-4D volume, dense", "mega": "mega volume 5D+, very dense"}
STYLE_DESC = {"natural": "even length across the lid", "cat": "cat-eye: longer toward the outer corner", "doll": "doll-eye: longest at the center"}
TEX_DESC = {"even": "evenly aligned tips", "wispy": "wispy: clusters gathered into pointed spikes with shorter lashes between them"}


def spec_text(m: dict) -> str:
    """번역된 래시 맵을 프롬프트용 사양 문장으로. 해석 여지를 줄이는 용도."""
    if not m:
        return ""
    z = m.get("zone_lengths_mm") or []
    parts = [
        f"Measured spec of image 2 (follow it exactly): {CURL_DESC.get(m.get('curl'), m.get('curl'))}",
        f"lengths from inner corner to outer corner: {', '.join(str(x) for x in z)} mm" if len(z) == 5 else "",
        f"density: {VOL_DESC.get(m.get('volume'), m.get('volume'))}",
        f"shape: {STYLE_DESC.get(m.get('style'), m.get('style'))}",
        f"texture: {TEX_DESC.get(m.get('texture'), m.get('texture'))}",
        f"notes: {m['notes']}" if m.get("notes") else "",
    ]
    return "\n" + "; ".join(x for x in parts if x) + "."


def split_data_url(data_url: str):
    header, b64 = data_url.split(",", 1)
    return header[5:].split(";")[0], b64


def gemini_tryon(frame_b64: str, frame_mime: str, ref_b64: str, ref_mime: str, prompt: str) -> tuple[str, str]:
    """Returns (mime_type, base64) of the edited image."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY가 없어요. spikes/a-live-ar/.env 에 GEMINI_API_KEY=... 를 넣고 서버를 다시 띄우세요. (Google AI Studio에서 발급)")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_IMAGE_MODEL}:generateContent"
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": frame_mime, "data": frame_b64}},
                {"inlineData": {"mimeType": ref_mime, "data": ref_b64}},
            ],
        }],
        "generationConfig": {"responseModalities": ["IMAGE"]},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json", "x-goog-api-key": key},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            resp = json.loads(r.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:600]
        raise RuntimeError(f"Gemini {e.code}: {detail}") from None
    # 응답에서 첫 이미지 파트 찾기 (inlineData / inline_data 둘 다 대응)
    for cand in resp.get("candidates", []):
        for part in cand.get("content", {}).get("parts", []):
            blob = part.get("inlineData") or part.get("inline_data")
            if blob and blob.get("data"):
                return blob.get("mimeType") or blob.get("mime_type") or "image/png", blob["data"]
    # 이미지가 없으면 텍스트라도 돌려줌 (거절 사유 등)
    texts = [p.get("text") for c in resp.get("candidates", []) for p in c.get("content", {}).get("parts", []) if p.get("text")]
    raise RuntimeError("이미지가 안 돌아왔어요: " + (" ".join(texts)[:300] or json.dumps(resp)[:300]))


def openai_tryon(frame_b64: str, frame_mime: str, ref_b64: str, ref_mime: str, prompt: str) -> tuple[str, str]:
    """OpenAI Images edit: image[] 여러 장 + prompt. input_fidelity=high 로 얼굴 보존."""
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 없어요. spikes/a-live-ar/.env 에 OPENAI_API_KEY=sk-... 를 넣고 서버를 다시 띄우세요.")
    ext = {"image/jpeg": "jpg", "image/png": "png", "image/webp": "webp"}
    files = [
        ("image[]", ("frame." + ext.get(frame_mime, "png"), base64.b64decode(frame_b64), frame_mime)),
        ("image[]", ("reference." + ext.get(ref_mime, "png"), base64.b64decode(ref_b64), ref_mime)),
    ]
    data = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": prompt + "\n(The first image is image 1, the second image is image 2.)",
        "size": "1536x1024",
        "quality": "high",
        "input_fidelity": "high",
        "n": "1",
    }
    r = requests.post("https://api.openai.com/v1/images/edits", headers={"Authorization": f"Bearer {key}"}, files=files, data=data, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:600]}")
    out = r.json()["data"][0]
    if out.get("b64_json"):
        return "image/png", out["b64_json"]
    if out.get("url"):
        img = requests.get(out["url"], timeout=60).content
        return "image/png", base64.b64encode(img).decode()
    raise RuntimeError("이미지가 안 돌아왔어요: " + json.dumps(out)[:300])


ENGINES = {"gemini": gemini_tryon, "openai": openai_tryon}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _json(self, code, obj):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path == "/api/tryon":
            return self.do_tryon()
        if self.path == "/api/feedback":
            return self.do_feedback()
        if self.path == "/api/eyepoints":
            n = int(self.headers.get("Content-Length", 0))
            try:
                req = json.loads(self.rfile.read(n))
                mt, b64 = split_data_url(req["image"])
                out = eyepoints(b64, mt)
                sys.stderr.write("eyepoints: %s\n" % json.dumps(out))
                return self._json(200, out)
            except Exception as e:  # noqa: BLE001
                return self._json(500, {"error": str(e)})
        if self.path != "/api/lashmap":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n))
            data_url = req["image"]  # data:image/jpeg;base64,....
            header, b64 = data_url.split(",", 1)
            media_type = header[5:].split(";")[0]
            if media_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
                return self._json(400, {"error": f"unsupported media type {media_type}"})
            result = translate(b64, media_type)
            sys.stderr.write("lashmap: %s\n" % json.dumps(result, ensure_ascii=False))
            return self._json(200, result)
        except requests.ConnectionError as e:
            return self._json(502, {"error": f"연결 실패: {e}"})
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("lashmap: ERR %s\n" % e)
            return self._json(500, {"error": str(e)})


    def do_tryon(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n))
            fm, fb = split_data_url(req["frame"])
            rm, rb = split_data_url(req["reference"])
            engine = req.get("engine", "gemini")
            if engine not in ENGINES:
                return self._json(400, {"error": f"unknown engine {engine}"})
            prompt = TRYON_PROMPT + spec_text(req.get("lashmap"))
            t0 = time.time()
            mime, b64 = ENGINES[engine](fb, fm, rb, rm, prompt)
            run_id = time.strftime("%Y%m%d-%H%M%S")
            # 실험 기록: 입력/출력 이미지 + 메타 (runs/ 는 gitignore)
            d = os.path.join(HERE, "runs", run_id); os.makedirs(d, exist_ok=True)
            for name, mt, data in (("frame", fm, fb), ("reference", rm, rb), ("result", mime, b64)):
                with open(os.path.join(d, name + "." + mt.split("/")[-1].replace("jpeg", "jpg")), "wb") as f:
                    f.write(base64.b64decode(data))
            meta = {"run_id": run_id, "engine": engine, "model": OPENAI_IMAGE_MODEL if engine == "openai" else GEMINI_IMAGE_MODEL,
                    "seconds": round(time.time() - t0, 1), "lashmap": req.get("lashmap"), "prompt": prompt}
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
            sys.stderr.write("tryon[%s] %s: ok %s %d bytes %.1fs\n" % (engine, run_id, mime, len(b64), meta["seconds"]))
            return self._json(200, {"image": f"data:{mime};base64,{b64}", "run_id": run_id, "seconds": meta["seconds"]})
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("tryon: ERR %s\n" % e)
            return self._json(500, {"error": str(e)})


    def do_feedback(self):
        """합성 결과 평가 저장: runs/<run_id>/meta.json 에 합치고 runs/log.jsonl 에 한 줄 추가."""
        n = int(self.headers.get("Content-Length", 0))
        try:
            fb = json.loads(self.rfile.read(n))
            run_id = fb.get("run_id", "")
            d = os.path.join(HERE, "runs", run_id)
            if not run_id or not os.path.isdir(d):
                return self._json(404, {"error": "run not found"})
            with open(os.path.join(d, "meta.json")) as f:
                meta = json.load(f)
            meta["feedback"] = {k: fb.get(k) for k in ("looks_like_me", "matches_reference", "note")}
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
            row = {k: meta.get(k) for k in ("run_id", "engine", "model", "seconds")}
            row.update(meta["feedback"]); row["lashmap"] = meta.get("lashmap")
            with open(os.path.join(HERE, "runs", "log.jsonl"), "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            return self._json(200, {"ok": True})
        except Exception as e:  # noqa: BLE001
            return self._json(500, {"error": str(e)})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    print(f"spike A/B on http://localhost:{port}  (openai key: {'yes' if os.environ.get('OPENAI_API_KEY') else 'NO'} [vision {OPENAI_VISION_MODEL}], gemini: {'yes' if os.environ.get('GEMINI_API_KEY') else 'NO'} [{GEMINI_IMAGE_MODEL}], openai: {'yes' if os.environ.get('OPENAI_API_KEY') else 'NO'} [{OPENAI_IMAGE_MODEL}])", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
