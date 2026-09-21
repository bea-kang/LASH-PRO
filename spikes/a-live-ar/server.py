"""spike A dev server: static files + POST /api/lashmap (reference photo -> lash map JSON).

Run:  python3 server.py  (port 8787)
Key:  같은 폴더의 .env 에 ANTHROPIC_API_KEY=sk-ant-... 한 줄 (gitignore됨), 또는 환경변수.
"""
import base64
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import urllib.error
import urllib.request

import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "claude-opus-5"
GEMINI_IMAGE_MODEL = os.environ.get("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image")


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
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["is_lash_photo", "curl", "len_inner_mm", "len_inner_mid_mm", "len_center_mm", "len_outer_mid_mm", "len_outer_mm", "volume", "style", "texture", "confidence", "notes"],
    "additionalProperties": False,
}

PROMPT = """이 사진은 속눈썹 연장 샵의 시술 결과 레퍼런스입니다. 이 디자인을 다른 사람 눈에 재현하기 위한 래시 맵(시술 사양)으로 번역하세요.

- curl: J(거의 직선, 살짝 올라감) / C(표준 컬) / D(강한 컬, 눈꺼풀 쪽으로 많이 감김)
- len_inner_mm → len_inner_mid_mm → len_center_mm → len_outer_mid_mm → len_outer_mm: 눈앞(내안각)부터 눈꼬리(외안각)까지 5구간의 연장 길이(mm). 실제 샵 사양 범위(6~15mm)로. 사진의 자연 속눈썹 길이와 비교해 추정.
- volume: natural(1:1 클래식, 성근 느낌) / volume(2~4D, 촘촘) / mega(5D 이상, 매우 빽빽)
- style: natural(균일) / cat(눈꼬리로 갈수록 길어짐) / doll(중앙이 가장 김)
- texture: even(길이가 고르게 정렬된 클래식/볼륨) / wispy(몇 가닥씩 뾰족하게 모여 스파이크를 이루는 요정·위스피 스타일. 긴 스파이크와 짧은 베이스가 번갈아 보임)
- confidence: 0~1. 사진이 눈을 감고 있거나 접사라 열린 눈 상태를 추정해야 하면 낮게.
- notes: 한 줄. 특징(예: "눈꼬리 3구간만 길게, 컬러 브라운") 또는 추정 근거.
- 속눈썹 사진이 아니면 is_lash_photo=false 로 하고 나머지는 기본값(C, 8-9-10-10-9, natural, natural, even, 0).
"""


def translate(image_b64: str, media_type: str) -> dict:
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_b64}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        output_config={"format": {"type": "json_schema", "schema": LASHMAP_SCHEMA}},
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("model refused")
    text = next(b.text for b in resp.content if b.type == "text")
    data = json.loads(text)
    data["zone_lengths_mm"] = [
        max(6, min(15, int(data.pop(k))))
        for k in ("len_inner_mm", "len_inner_mid_mm", "len_center_mm", "len_outer_mid_mm", "len_outer_mm")
    ]
    data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))
    data["_usage"] = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
    return data


TRYON_PROMPT = """Image 1 is a close-up photo of a person's eyes. Image 2 is a reference photo of eyelash extensions done at a lash salon.
Edit image 1 so the person is wearing eyelash extensions that reproduce image 2 as faithfully as possible: the same curl, the same lengths per zone (inner corner to outer corner), the same density, the same spiky/wispy or even pattern, the same lash color.
Keep everything else in image 1 exactly as it is: eye shape and size, eyelids, skin texture, iris, eyebrows, lighting, framing, resolution. Do not enlarge the eyes, do not retouch the skin, do not add eyeliner or other makeup, do not change the face.
Return only the edited version of image 1, same size and framing."""


def split_data_url(data_url: str):
    header, b64 = data_url.split(",", 1)
    return header[5:].split(";")[0], b64


def gemini_tryon(frame_b64: str, frame_mime: str, ref_b64: str, ref_mime: str) -> tuple[str, str]:
    """Returns (mime_type, base64) of the edited image."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY가 없어요. spikes/a-live-ar/.env 에 GEMINI_API_KEY=... 를 넣고 서버를 다시 띄우세요. (Google AI Studio에서 발급)")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_IMAGE_MODEL}:generateContent"
    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": TRYON_PROMPT},
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
        if self.path != "/api/lashmap":
            return self._json(404, {"error": "not found"})
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return self._json(500, {"error": "ANTHROPIC_API_KEY가 없어요. spikes/a-live-ar/.env 에 ANTHROPIC_API_KEY=sk-ant-... 를 넣고 서버를 다시 띄우세요."})
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
        except anthropic.AuthenticationError:
            return self._json(401, {"error": "API 키가 잘못됐어요."})
        except anthropic.RateLimitError:
            return self._json(429, {"error": "요청이 많아요. 잠시 후 다시."})
        except anthropic.APIStatusError as e:
            return self._json(e.status_code, {"error": f"API {e.status_code}: {e.message}"})
        except anthropic.APIConnectionError as e:
            return self._json(502, {"error": f"연결 실패: {e}"})
        except Exception as e:  # noqa: BLE001
            return self._json(500, {"error": str(e)})


    def do_tryon(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n))
            fm, fb = split_data_url(req["frame"])
            rm, rb = split_data_url(req["reference"])
            mime, b64 = gemini_tryon(fb, fm, rb, rm)
            sys.stderr.write("tryon: ok %s %d bytes\n" % (mime, len(b64)))
            return self._json(200, {"image": f"data:{mime};base64,{b64}"})
        except Exception as e:  # noqa: BLE001
            sys.stderr.write("tryon: ERR %s\n" % e)
            return self._json(500, {"error": str(e)})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    print(f"spike A/B on http://localhost:{port}  (anthropic key: {'yes' if os.environ.get('ANTHROPIC_API_KEY') else 'NO'}, gemini key: {'yes' if os.environ.get('GEMINI_API_KEY') else 'NO'}, image model: {GEMINI_IMAGE_MODEL})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
