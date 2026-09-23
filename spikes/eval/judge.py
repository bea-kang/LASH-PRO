"""레퍼런스 vs 결과 유사도 — 비전 모델 심사 (LLM-as-judge).

레퍼런스 눈 크롭과 결과 눈 크롭을 같이 보여주고 항목별 1~5 채점. 같은 입력을 N번 반복해 평균.
사용: python3 judge.py <reference.jpg> <result.png> [--crop-result] [--n 3]
"""
import base64
import io
import json
import os
import sys

import requests
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "a-live-ar"))
import server  # noqa: E402  (.env)

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.5")
SCHEMA = {
    "type": "object",
    "properties": {
        "length_match": {"type": "integer", "enum": [1, 2, 3, 4, 5]},      # 길이(눈 대비 비율)
        "shape_match": {"type": "integer", "enum": [1, 2, 3, 4, 5]},       # 구간별 길이 분포(캣/돌/균일)
        "texture_match": {"type": "integer", "enum": [1, 2, 3, 4, 5]},     # 스파이크/위스피 vs 고른 정렬, 스파이크 수
        "density_match": {"type": "integer", "enum": [1, 2, 3, 4, 5]},     # 밀도·진하기
        "lower_lash_match": {"type": "integer", "enum": [1, 2, 3, 4, 5]},  # 아래 속눈썹 유무·정도
        "realism": {"type": "integer", "enum": [1, 2, 3, 4, 5]},           # 결과가 실제 시술 사진처럼 보이는가
        "overall_similarity": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "biggest_difference": {"type": "string"},
    },
    "required": ["length_match", "shape_match", "texture_match", "density_match", "lower_lash_match", "realism", "overall_similarity", "biggest_difference"],
    "additionalProperties": False,
}
PROMPT = """You are a lash extension technician judging a virtual try-on.
Image 1 is the salon REFERENCE design. Image 2 is a RESULT: the same design supposedly applied to a different person's eye.
Judge how faithfully the RESULT reproduces the REFERENCE design (not whether it looks pretty). Different eye shapes are expected; judge the lash design itself, relative to each eye's size.
Score 1-5 (5 = identical):
- length_match: longest lashes relative to eye width.
- shape_match: where the lashes are longest (inner / center / outer) and how length changes along the lid.
- texture_match: spiky/wispy clusters vs evenly aligned tips; number and sharpness of spikes.
- density_match: how full and dark the lash line is.
- lower_lash_match: whether lower lashes are extended as in the reference (5 if both have none or both match).
- realism: does the RESULT look like a real photo of real lash extensions (not drawn/pasted).
- overall_similarity: overall, how close the design is.
- biggest_difference: one sentence, the single biggest difference."""


def b64(img, max_side=1024):
    k = min(1, max_side / max(img.size)); img = img.resize((int(img.width * k), int(img.height * k)))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=90); return base64.b64encode(buf.getvalue()).decode()


def judge(ref_img, out_img, n=3):
    key = os.environ["OPENAI_API_KEY"]
    rb, ob = b64(ref_img), b64(out_img)
    runs = []
    for _ in range(n):
        body = {
            "model": JUDGE_MODEL, "reasoning_effort": "medium",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "Image 1 (reference):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{rb}", "detail": "high"}},
                {"type": "text", "text": "Image 2 (result):"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ob}", "detail": "high"}},
                {"type": "text", "text": PROMPT},
            ]}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "lash_judge", "strict": True, "schema": SCHEMA}},
        }
        r = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=180)
        if r.status_code != 200:
            raise RuntimeError(f"OpenAI {r.status_code}: {r.text[:300]}")
        runs.append(json.loads(r.json()["choices"][0]["message"]["content"]))
    keys = [k for k in SCHEMA["properties"] if k != "biggest_difference"]
    avg = {k: round(sum(x[k] for x in runs) / n, 2) for k in keys}
    return {"avg": avg, "runs": runs}


def left_eye_crop(img, frac=0.5):
    return img.crop((0, 0, int(img.width * frac), img.height))


if __name__ == "__main__":
    ref = Image.open(sys.argv[1]).convert("RGB"); out = Image.open(sys.argv[2]).convert("RGB")
    if "--crop-result" in sys.argv: out = left_eye_crop(out)
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 3
    res = judge(ref, out, n)
    print(json.dumps(res, ensure_ascii=False, indent=1))
