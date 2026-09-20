"""spike A dev server: static files + POST /api/lashmap (reference photo -> lash map JSON).

Run:  ANTHROPIC_API_KEY=... python3 server.py  (port 8787)
"""
import base64
import json
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import anthropic

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "claude-opus-5"

LASHMAP_SCHEMA = {
    "type": "object",
    "properties": {
        "is_lash_photo": {"type": "boolean"},
        "curl": {"type": "string", "enum": ["J", "C", "D"]},
        # structured output은 minItems>1을 안 받아서 5구간을 개별 필드로
        "len_inner_mm": {"type": "integer", "minimum": 6, "maximum": 15},
        "len_inner_mid_mm": {"type": "integer", "minimum": 6, "maximum": 15},
        "len_center_mm": {"type": "integer", "minimum": 6, "maximum": 15},
        "len_outer_mid_mm": {"type": "integer", "minimum": 6, "maximum": 15},
        "len_outer_mm": {"type": "integer", "minimum": 6, "maximum": 15},
        "volume": {"type": "string", "enum": ["natural", "volume", "mega"]},
        "style": {"type": "string", "enum": ["natural", "cat", "doll"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "notes": {"type": "string"},
    },
    "required": ["is_lash_photo", "curl", "len_inner_mm", "len_inner_mid_mm", "len_center_mm", "len_outer_mid_mm", "len_outer_mm", "volume", "style", "confidence", "notes"],
    "additionalProperties": False,
}

PROMPT = """이 사진은 속눈썹 연장 샵의 시술 결과 레퍼런스입니다. 이 디자인을 다른 사람 눈에 재현하기 위한 래시 맵(시술 사양)으로 번역하세요.

- curl: J(거의 직선, 살짝 올라감) / C(표준 컬) / D(강한 컬, 눈꺼풀 쪽으로 많이 감김)
- len_inner_mm → len_inner_mid_mm → len_center_mm → len_outer_mid_mm → len_outer_mm: 눈앞(내안각)부터 눈꼬리(외안각)까지 5구간의 연장 길이(mm). 실제 샵 사양 범위(6~15mm)로. 사진의 자연 속눈썹 길이와 비교해 추정.
- volume: natural(1:1 클래식, 성근 느낌) / volume(2~4D, 촘촘) / mega(5D 이상, 매우 빽빽)
- style: natural(균일) / cat(눈꼬리로 갈수록 길어짐) / doll(중앙이 가장 김)
- confidence: 사진이 눈을 감고 있거나 접사라 열린 눈 상태를 추정해야 하면 낮게.
- notes: 한 줄. 특징(예: "눈꼬리 3구간만 길게, 컬러 브라운") 또는 추정 근거.
- 속눈썹 사진이 아니면 is_lash_photo=false 로 하고 나머지는 기본값(C, 8-9-10-10-9, natural, natural, 0).
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
    data["zone_lengths_mm"] = [data.pop(k) for k in ("len_inner_mm", "len_inner_mid_mm", "len_center_mm", "len_outer_mid_mm", "len_outer_mm")]
    data["_usage"] = {"in": resp.usage.input_tokens, "out": resp.usage.output_tokens}
    return data


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
        if self.path != "/api/lashmap":
            return self._json(404, {"error": "not found"})
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return self._json(500, {"error": "ANTHROPIC_API_KEY가 없어요. 서버를 ANTHROPIC_API_KEY=... 로 실행하세요."})
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n))
            data_url = req["image"]  # data:image/jpeg;base64,....
            header, b64 = data_url.split(",", 1)
            media_type = header[5:].split(";")[0]
            if media_type not in ("image/jpeg", "image/png", "image/webp", "image/gif"):
                return self._json(400, {"error": f"unsupported media type {media_type}"})
            result = translate(b64, media_type)
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


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
    print(f"spike A on http://localhost:{port}  (key: {'yes' if os.environ.get('ANTHROPIC_API_KEY') else 'NO'})", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
