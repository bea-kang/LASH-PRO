"""spike dev server: static files + /api/lashmap (사진 -> 래시 맵 JSON) + /api/tryon (사진 그대로 합성).

Run:  python3 server.py  (port 8787)
Keys: 같은 폴더의 .env (gitignore됨). OPENAI_API_KEY 하나로 다 됨. GEMINI_API_KEY 는 비교용 선택.
"""
import base64
import json
import os
import ssl
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import urllib.error
import urllib.request

import requests

sys.path.insert(0, os.path.join(HERE, "..", "eval")) if False else None

HERE = os.path.dirname(os.path.abspath(__file__))
OPENAI_VISION_MODEL = os.environ.get("OPENAI_VISION_MODEL", "gpt-4.1")          # 1차: 래시 맵 + 대략 위치 (빠름)
OPENAI_POINTS_MODEL = os.environ.get("OPENAI_POINTS_MODEL", "gpt-5.5")          # 2차: 정밀 점 (2026-09-22 비교: 4.1보다 확실히 정확, low effort 8초)
OPENAI_POINTS_EFFORT = os.environ.get("OPENAI_POINTS_EFFORT", "low")
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
        # 시판 속눈썹(스트립) 제품 사진: 흰/단색 배경에 띠 형태. 이 경우 눈 기준점 대신 띠 양 끝점을 씀
        "is_product_strip": {"type": "boolean"},
        "strip_left_x": {"type": "number"}, "strip_left_y": {"type": "number"},
        "strip_right_x": {"type": "number"}, "strip_right_y": {"type": "number"},
        "strip_lashes_side": {"type": "string", "enum": ["up", "down", "left", "right"]},
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
        "lower_lashes": {"type": "boolean"},
        "drama": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        # 레퍼런스 눈 기준점 (이미지 폭·높이 대비 0~1). 눈이 여러 개면 가장 크고 선명한 눈 하나
        "inner_corner_x": {"type": "number"}, "inner_corner_y": {"type": "number"},
        "outer_corner_x": {"type": "number"}, "outer_corner_y": {"type": "number"},
        "lash_root_center_x": {"type": "number"}, "lash_root_center_y": {"type": "number"},
        "confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["is_lash_photo", "is_product_strip", "strip_left_x", "strip_left_y", "strip_right_x", "strip_right_y", "strip_lashes_side", "curl", "len_inner_mm", "len_inner_mid_mm", "len_center_mm", "len_outer_mid_mm", "len_outer_mm", "volume", "style", "texture", "lower_lashes", "drama", "inner_corner_x", "inner_corner_y", "outer_corner_x", "outer_corner_y", "lash_root_center_x", "lash_root_center_y", "confidence", "notes"],
    "additionalProperties": False,
}

PROMPT = """이 사진은 속눈썹 디자인 레퍼런스입니다. (a) 샵 시술 결과 사진이거나 (b) 시판 속눈썹(스트립/붙임 속눈썹) 제품 사진입니다. 이 디자인을 다른 사람 눈에 재현하기 위한 래시 맵(시술 사양)으로 번역하세요.

- is_product_strip: 눈이 아니라 제품 자체(띠에 붙은 속눈썹, 보통 흰/단색 배경, 케이스 위)가 찍힌 사진이면 true. 이 경우 strip_left/right는 띠(뿌리 밴드)의 양 끝점(이미지 폭·높이 대비 0~1), strip_lashes_side는 띠 기준으로 속눈썹 가닥이 뻗어 있는 방향. 제품이 여러 개면 가장 크고 선명한 것 하나. 눈 사진이면 false로 두고 strip 값은 0, side는 up.

- curl: J(거의 직선, 살짝 올라감) / C(표준 컬) / D(강한 컬, 눈꺼풀 쪽으로 많이 감김)
- len_inner_mm → len_inner_mid_mm → len_center_mm → len_outer_mid_mm → len_outer_mm: 눈앞(내안각)부터 눈꼬리(외안각)까지 5구간의 연장 길이(mm). 실제 샵 사양 범위(6~15mm)로. 사진의 자연 속눈썹 길이와 비교해 추정.
- volume: natural(1:1 클래식, 성근 느낌) / volume(2~4D, 촘촘) / mega(5D 이상, 매우 빽빽)
- style: natural(균일) / cat(눈꼬리로 갈수록 길어짐) / doll(중앙이 가장 김)
- texture: even(길이가 고르게 정렬된 클래식/볼륨) / wispy(몇 가닥씩 뾰족하게 모여 스파이크를 이루는 요정·위스피 스타일. 긴 스파이크와 짧은 베이스가 번갈아 보임)
- lower_lashes: 아래 속눈썹에도 연장/스파이크가 있으면 true.
- drama: 과감도 1(아주 자연스러움)~5(매우 과감, 길고 빽빽). 샵 레퍼런스는 대개 3~5. 실제보다 낮게 읽는 경향이 있으니 보이는 그대로.
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


EYEPOINTS_EXT_SCHEMA = json.loads(json.dumps(EYEPOINTS_SCHEMA))
for _j in range(5):
    EYEPOINTS_EXT_SCHEMA["properties"][f"lower_x{_j}"] = {"type": "number"}; EYEPOINTS_EXT_SCHEMA["properties"][f"lower_y{_j}"] = {"type": "number"}
    EYEPOINTS_EXT_SCHEMA["required"] += [f"lower_x{_j}", f"lower_y{_j}"]
EYEPOINTS_EXT_PROMPT = EYEPOINTS_PROMPT + "- lower_0..lower_4: 아랫눈꺼풀 라인(눈 구멍의 아래 경계, 아래 속눈썹 뿌리)을 눈앞(lower_0)에서 눈꼬리(lower_4)까지 등간격 5점.\n"


def eyepoints(image_b64: str, media_type: str, lower: bool = False) -> dict:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY가 없어요.")
    body = {
        "model": OPENAI_POINTS_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_b64}", "detail": "high"}},
            {"type": "text", "text": EYEPOINTS_EXT_PROMPT if lower else EYEPOINTS_PROMPT},
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {"name": "eye_points", "strict": True, "schema": EYEPOINTS_EXT_SCHEMA if lower else EYEPOINTS_SCHEMA}},
    }
    if OPENAI_POINTS_MODEL.startswith(("gpt-5", "o")):
        body["reasoning_effort"] = OPENAI_POINTS_EFFORT
    r = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:600]}")
    msg = r.json()["choices"][0]["message"]
    if msg.get("refusal"):
        raise RuntimeError("model refused: " + msg["refusal"])
    return json.loads(msg["content"])


STYLE_LINE = {"natural": "even length across the lid", "cat": "cat-eye: longest toward the outer corner, sweeping outward", "doll": "doll-eye: longest at the center of the lid"}
TEX_LINE = {"even": "evenly aligned tips (classic/volume set)", "wispy": "wispy: distinct pointed clusters (spikes) with shorter lashes between them"}
CURL_LINE = {"J": "J curl (nearly straight, slight lift)", "C": "C curl, lifted upward", "D": "D curl, strongly curled up toward the lid"}


def build_prompt(m, geom, guided):
    """레퍼런스 충실도 프롬프트. 2026-09-22 비교: 추정 mm 사양은 결과를 얌전하게 만들어 뺌. 실측 기하(픽셀)만 숫자로 넣음."""
    m = m or {}; geom = geom or {}
    lines = []
    if guided:
        lines.append("Image 1 is a close-up photo of a person's eyes with a ROUGH OVERLAY of the eyelash extensions already drawn in place (dark strands on the upper lash line). Image 2 is the salon reference photo of the same lash design.")
        lines.append("Turn image 1 into a photorealistic photo of this person actually wearing these eyelash extensions. The overlay marks the correct placement, lengths and spike positions — keep them there; replace the flat drawn strands with realistic individual lashes matching image 2's texture, sheen and color.")
    else:
        lines.append("Image 1 is a close-up photo of a person's eyes. Image 2 is a salon reference photo of eyelash extensions.")
        lines.append("Put the EXACT lash design from image 2 onto the person in image 1.")
    drama = m.get("drama") or 3
    sparse = m.get("volume") == "natural" or m.get("texture") == "wispy"
    if sparse:
        lines.append("IMPORTANT: image 2 is a SPARSE, separated design — only a few distinct lashes/spikes with visible skin and gaps between them. Do NOT fill the lash line into a full, dense, uniform fan. Fewer lashes than you would normally draw. If in doubt, draw fewer.")
    if drama >= 4:
        lines.append("Match image 2's boldness (long, dramatic) at the same intensity relative to the eye. Do not shorten it.")
    else:
        lines.append("Match image 2's intensity exactly — neither bolder nor more natural than the reference.")
    lines.append("Copy these from image 2 precisely:")
    tip = geom.get("tipRatio")
    if tip:
        lines.append(f"- Length: the longest lashes reach about {int(round(tip * 100))}% of the eye's width above the lash line (measured on image 2). Keep that proportion on image 1's eyes.")
    spikes = geom.get("spikes")
    if m.get("texture") == "wispy":
        lines.append(f"- Spikes: {TEX_LINE['wispy']}" + (f", about {spikes} spikes per upper lid" if spikes else "") + ".")
    else:
        lines.append(f"- Tips: {TEX_LINE.get(m.get('texture'), 'as in image 2')}.")
    lines.append(f"- Shape: {STYLE_LINE.get(m.get('style'), 'as in image 2')}.")
    lines.append("- Density: exactly as dense as image 2 — no fuller." if sparse else "- Density: full and dark along the whole upper lash line, as dense as image 2.")
    if m.get("lower_lashes"):
        lines.append("- Lower lashes: image 2 also has short wispy spikes on the lower lash line — add them too.")
    lines.append(f"- Curl: {CURL_LINE.get(m.get('curl'), 'as in image 2')}. Color: black.")
    lines.append("Do not reinterpret the style. Keep everything else in image 1 exactly as it is: eye shape and size, eyelids, skin, iris, eyebrows, lighting, framing. No skin retouching, no eyeliner, no other makeup, no face changes.")
    lines.append("Return only the edited version of image 1, same size and framing.")
    return "\n".join(lines)


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


def gemini_tryon(frame_b64, frame_mime, ref_b64, ref_mime, prompt):
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
        "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": {"aspectRatio": "3:2"}},  # 입력(1536x1024)과 같은 비율로 → 전/후 정렬
    }
    def _call(b):
        req = urllib.request.Request(url, data=json.dumps(b).encode(), method="POST", headers={"Content-Type": "application/json", "x-goog-api-key": key})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    try:
        try:
            resp = _call(body)
        except urllib.error.HTTPError as e:
            if e.code == 400 and "imageConfig" in json.dumps(body):   # 모델이 imageConfig 를 안 받으면 빼고 재시도
                body["generationConfig"].pop("imageConfig", None); resp = _call(body)
            else:
                raise
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


def openai_tryon(frame_b64, frame_mime, ref_b64, ref_mime, prompt):
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


# ---- 눈 단위 합성 (2026-09-26): 배율 맞춘 눈 크롭 → 생성 → 속눈썹만 전이 (lashcomp.py) ----
def single_eye_prompt(prompt: str) -> str:
    """전체 프레임용 프롬프트를 한쪽 눈 크롭용으로. 배율이 같다는 것과 눈을 옮기지 말라는 것을 명시."""
    prompt = prompt.replace("Image 1 is a close-up photo of a person's eyes. Image 2 is a salon reference photo of eyelash extensions.",
                            "Image 1 is a close-up photo of ONE eye of a person. Image 2 is a salon reference photo of eyelash extensions on one eye, shown at the SAME SCALE as image 1 (both eyes are about the same width in the frame), so copy the lash lengths and thickness at the same pixel scale.")
    return prompt.replace("Return only the edited version of image 1, same size and framing.",
                          "The eye in image 1 must stay at exactly the same position, size and shape; only add the lashes. Return only the edited version of image 1, same size and framing.")


def _pts_to_eye(m: dict, W: int, H: int) -> dict:
    P = lambda kx, ky: (m[kx] * W, m[ky] * H)
    e = {"inner": P("inner_corner_x", "inner_corner_y"), "outer": P("outer_corner_x", "outer_corner_y"), "root": [P(f"root_x{j}", f"root_y{j}") for j in range(5)]}
    if "lower_x0" in m: e["lower"] = [P(f"lower_x{j}", f"lower_y{j}") for j in range(5)]
    return e


def tryon_eyes(frame_b64, ref_b64, ref_mime, eyes, ref_eye, engine, prompt, lower, both=False):
    """eyes: {side: {inner:[x,y], outer:[x,y], root:[[x,y]*n], lower:[[x,y]*n]}} (프레임 픽셀). ref_eye: {inner, outer} (레퍼런스 크롭 픽셀, 1536x1024 기준).
    returns (result PIL, per-eye dict)."""
    import io as _io, threading
    import lashcomp
    from PIL import Image
    frame = Image.open(_io.BytesIO(base64.b64decode(frame_b64))).convert("RGB"); W, H = frame.size
    ri, ro = ref_eye["inner"], ref_eye["outer"]
    ref_frac = max(0.2, min(0.8, ((ro[0] - ri[0]) ** 2 + (ro[1] - ri[1]) ** 2) ** 0.5 / lashcomp.OUT_W))
    ref_center = ((ri[0] + ro[0]) / 2 / lashcomp.OUT_W, (ri[1] + ro[1]) / 2 / lashcomp.OUT_H)
    per = {}
    geo = {}
    for side, e in eyes.items():
        eye = {"inner": tuple(e["inner"]), "outer": tuple(e["outer"]), "root": [tuple(p) for p in e["root"]]}
        if e.get("lower"): eye["lower"] = [tuple(p) for p in e["lower"]]
        geo[side] = (eye, lashcomp.crop_box(eye, W, H, ref_frac, ref_center))
    # 기본은 한쪽 눈만 생성 (더 큰/정면인 눈) → (u,v) 매트를 두 눈에 렌더. 두 눈 따로 생성하면 길이·밀도가 서로 달라져 "비율이 흔들림" (2026-09-26 폰 테스트)
    gen_sides = list(eyes) if both else [max(geo, key=lambda k: lashcomp.eye_axes(geo[k][0])[3])]
    def work(side, e):
        eye, box = geo[side]
        cb = lashcomp.crop_b64(frame, box)
        t0 = time.time()
        mime, out_b64 = ENGINES[engine](cb, "image/jpeg", ref_b64, ref_mime, prompt)
        t1 = time.time()
        out = Image.open(_io.BytesIO(base64.b64decode(out_b64))).convert("RGB").resize((lashcomp.OUT_W, lashcomp.OUT_H), Image.LANCZOS)
        buf = _io.BytesIO(); out.save(buf, "JPEG", quality=90); out_jpg = base64.b64encode(buf.getvalue()).decode()
        m = eyepoints(out_jpg, "image/jpeg", lower=True)           # 생성 결과 위의 눈 점 (눈이 옮겨져도 (u,v) 로 전이)
        sys.stderr.write("tryon_eyes[%s]: gen %.1fs (%s) · points %.1fs\n" % (side, t1 - t0, GEMINI_IMAGE_MODEL if engine == "gemini" else OPENAI_IMAGE_MODEL, time.time() - t1))
        eye_out = _pts_to_eye(m, lashcomp.OUT_W, lashcomp.OUT_H)
        per[side] = {"eye": eye, "box": box, "out": out, "out_jpg": out_jpg, "pts": m, "eye_out": eye_out, "gen_seconds": round(time.time() - t0, 1)}
    threads = [threading.Thread(target=work, args=(side, eyes[side])) for side in gen_sides]
    for t in threads: t.start()
    for t in threads: t.join()
    if len(per) != len(gen_sides):
        raise RuntimeError("눈 합성 실패 (서버 로그 참고)")
    result = frame; t2 = time.time()
    for side in eyes:
        src = per[side] if side in per else per[gen_sides[0]]          # 생성 안 한 눈은 생성한 눈의 매트를 그대로
        maps = src.setdefault("maps", lashcomp.lash_maps(src["out"], src["eye_out"], lower))
        eye, box = geo[side]
        result, a_mean = lashcomp.render_eye(result, eye, box, maps)
        per.setdefault(side, {"eye": eye, "box": box, "from": gen_sides[0]})["alpha_mean"] = round(a_mean, 4)
    sys.stderr.write("tryon_eyes: render %.1fs\n" % (time.time() - t2))
    return result, per


def judge_result(ref_b64, ref_mime, out_b64, out_mime, crop_left=True):
    """합성 결과를 레퍼런스와 비교 채점 (spikes/eval/judge.py). 실패해도 합성은 성공으로 둠."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("judge", os.path.join(HERE, "..", "eval", "judge.py"))
        judge = importlib.util.module_from_spec(spec); spec.loader.exec_module(judge)
        from PIL import Image
        import io as _io
        ref = Image.open(_io.BytesIO(base64.b64decode(ref_b64))).convert("RGB")
        out = Image.open(_io.BytesIO(base64.b64decode(out_b64))).convert("RGB")
        # 결과는 양쪽 눈 크롭(1536x1024) → 왼쪽 절반(한쪽 눈)만
        if crop_left: out = out.crop((0, 0, out.width // 2, out.height))
        return judge.judge(ref, out, n=1)["runs"][0]
    except Exception as e:  # noqa: BLE001
        sys.stderr.write("judge: ERR %s\n" % e)
        return None


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
            engine = req.get("engine", "auto")
            lm = req.get("lashmap") or {}
            if engine == "auto":
                # 2026-09-26 비교: 위스피/과감 → gemini-3-pro-image(질감 4.0), 고른 스타일 → gemini-3.1-flash-image(실제감 5.0, 싸다)
                if os.environ.get("GEMINI_API_KEY"):
                    engine = "gemini"
                    global GEMINI_IMAGE_MODEL
                    GEMINI_IMAGE_MODEL = "gemini-3-pro-image" if (lm.get("texture") == "wispy" or (lm.get("drama") or 0) >= 4) else "gemini-3.1-flash-image"
                else:
                    engine = "openai"
            if engine not in ENGINES:
                return self._json(400, {"error": f"unknown engine {engine}"})
            prompt = build_prompt(req.get("lashmap"), req.get("geom"), bool(req.get("guided")))
            t0 = time.time()
            run_id = time.strftime("%Y%m%d-%H%M%S")
            d = os.path.join(HERE, "runs", run_id); os.makedirs(d, exist_ok=True)
            extra = {}
            if req.get("eyes"):
                # 눈 단위: 눈마다 레퍼런스 배율로 크롭 → 생성 → 속눈썹만 전이. 눈·피부·눈썹은 원본 그대로
                import io as _io
                prompt = single_eye_prompt(prompt)
                lower = bool(lm.get("lower_lashes"))
                result, per = tryon_eyes(fb, rb, rm, req["eyes"], req["refEye"], engine, prompt, lower, both=bool(req.get("both")))
                buf = _io.BytesIO(); result.save(buf, "JPEG", quality=92); mime, b64 = "image/jpeg", base64.b64encode(buf.getvalue()).decode()
                for side, e in per.items():
                    if "out" in e:
                        e["out"].save(os.path.join(d, f"eye-{side}-raw.jpg"), quality=90)
                        extra.setdefault("eyes", {})[side] = {"box": e["box"], "out": "data:image/jpeg;base64," + e["out_jpg"], "pts": e["pts"], "gen_seconds": e["gen_seconds"], "alpha_mean": e.get("alpha_mean")}
                    else:
                        extra.setdefault("eyes", {})[side] = {"box": e["box"], "from": e["from"], "alpha_mean": e.get("alpha_mean")}
                # 심사용: 생성한 눈 박스를 결과에서 잘라 레퍼런스와 비교
                side0 = next(k for k, v in per.items() if "out" in v); x0, y0, cw, ch = per[side0]["box"]
                jb = _io.BytesIO(); result.crop((x0, y0, x0 + cw, y0 + ch)).resize((1536, 1024)).save(jb, "JPEG", quality=90)
                judge_b64, judge_mime, judge_crop = base64.b64encode(jb.getvalue()).decode(), "image/jpeg", False
            else:
                mime, b64 = ENGINES[engine](fb, fm, rb, rm, prompt)
                judge_b64, judge_mime, judge_crop = b64, mime, True
            # 실험 기록: 입력/출력 이미지 + 메타 (runs/ 는 gitignore)
            for name, mt, data in (("frame", fm, fb), ("reference", rm, rb), ("result", mime, b64)):
                with open(os.path.join(d, name + "." + mt.split("/")[-1].replace("jpeg", "jpg")), "wb") as f:
                    f.write(base64.b64decode(data))
            meta = {"run_id": run_id, "engine": engine, "mode": "eyes" if req.get("eyes") else "frame", "guided": bool(req.get("guided")), "model": OPENAI_IMAGE_MODEL if engine == "openai" else GEMINI_IMAGE_MODEL,
                    "seconds": round(time.time() - t0, 1), "lashmap": req.get("lashmap"), "geom": req.get("geom"), "prompt": prompt,
                    "eyes": {k: {"box": v["box"], "pts": v.get("pts"), "from": v.get("from"), "alpha_mean": v.get("alpha_mean"), "gen_seconds": v.get("gen_seconds")} for k, v in extra.get("eyes", {}).items()} or None}
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(meta, f, ensure_ascii=False, indent=1)
            sys.stderr.write("tryon[%s/%s] %s: ok %s %d bytes %.1fs\n" % (engine, meta["mode"], run_id, mime, len(b64), meta["seconds"]))
            score = judge_result(rb, rm, judge_b64, judge_mime, crop_left=judge_crop) if req.get("judge", True) else None
            if score:
                meta["judge"] = score
                with open(os.path.join(d, "meta.json"), "w") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=1)
                sys.stderr.write("judge %s: overall %s tex %s len %s | %s\n" % (run_id, score["overall_similarity"], score["texture_match"], score["length_match"], score["biggest_difference"]))
            return self._json(200, {"image": f"data:{mime};base64,{b64}", "run_id": run_id, "seconds": meta["seconds"], "judge": score, **extra})
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
    # --https: 폰 테스트용 (카메라는 https에서만 켜짐). .cert/ 의 자체서명 인증서 사용, 폰에서 경고 → "계속" 필요
    https = "--https" in sys.argv
    srv = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    if https:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(os.path.join(HERE, ".cert", "cert.pem"), os.path.join(HERE, ".cert", "key.pem"))
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    print(f"spike A/B on {'https' if https else 'http'}://localhost:{port}  (openai key: {'yes' if os.environ.get('OPENAI_API_KEY') else 'NO'} [vision {OPENAI_VISION_MODEL}, points {OPENAI_POINTS_MODEL}/{OPENAI_POINTS_EFFORT}], gemini: {'yes' if os.environ.get('GEMINI_API_KEY') else 'NO'} [{GEMINI_IMAGE_MODEL}], openai: {'yes' if os.environ.get('OPENAI_API_KEY') else 'NO'} [{OPENAI_IMAGE_MODEL}])", flush=True)
    srv.serve_forever()
