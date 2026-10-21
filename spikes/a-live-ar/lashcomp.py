"""눈 단위 합성 파이프라인 (2026-09-26).

문제: 얼굴 전체 프레임(눈 = 폭의 ~20%)과 접사 레퍼런스(눈 = 폭의 ~45%)를 그대로 모델에 주면
(1) 배율이 안 맞아 길이·굵기가 어긋나고 (2) 모델이 눈 모양·피부까지 다시 그린다.

해결:
  1. 사용자 프레임에서 눈 하나를 레퍼런스와 같은 배율로 자른다 (눈 너비 / 크롭 너비 를 레퍼런스와 맞춤).
  2. 그 크롭 + 레퍼런스로 생성.
  3. 생성 결과를 입력 크롭에 정렬(평행이동)한 뒤, 속눈썹 존 안에서 "입력보다 어두워진 픽셀"만 원본에 얹는다.
     → 존 밖·밝아진 픽셀(피부 보정, 눈 모양 변경)은 전부 버려져 원본이 그대로 남는다.
  4. 원본 프레임 좌표로 되돌려 붙인다.

눈 기하(eye): {"inner":(x,y), "outer":(x,y), "root":[(x,y)*5], "lower":[(x,y)...] (선택)} — 프레임 픽셀 좌표.
"""
import io
import numpy as np
from PIL import Image, ImageFilter

OUT_W, OUT_H = 1536, 1024


def _np(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def _lum(a):
    return a[..., 0] * 0.299 + a[..., 1] * 0.587 + a[..., 2] * 0.114


def eye_axes(eye):
    inner, outer = np.array(eye["inner"], float), np.array(eye["outer"], float)
    root = [np.array(p, float) for p in eye["root"]]
    w = float(np.linalg.norm(outer - inner)) or 1.0
    axis = (outer - inner) / w
    perp = np.array([-axis[1], axis[0]])
    if np.dot(perp, root[2] - (inner + outer) / 2) < 0:
        perp = -perp                       # perp = 뿌리 라인에서 속눈썹(위) 쪽
    return inner, outer, root, w, axis, perp


def crop_box(eye, W, H, ref_frac=0.45, ref_center=(0.37, 0.47), ar=OUT_W / OUT_H):
    """눈 너비가 크롭 폭의 ref_frac 이 되도록, 눈 중심이 크롭의 ref_center 에 오도록 박스 계산 (프레임 픽셀)."""
    inner, outer, root, w, axis, perp = eye_axes(eye)
    cw = w / ref_frac; ch = cw / ar
    cw = min(cw, W); ch = min(ch, H)
    if cw / ch > ar: cw = ch * ar
    else: ch = cw / ar
    c = (inner + outer) / 2
    x0 = c[0] - cw * ref_center[0]; y0 = c[1] - ch * ref_center[1]
    x0 = max(0, min(W - cw, x0)); y0 = max(0, min(H - ch, y0))
    return int(round(x0)), int(round(y0)), int(round(cw)), int(round(ch))


def to_crop(eye, box, size=(OUT_W, OUT_H)):
    x0, y0, cw, ch = box; sx, sy = size[0] / cw, size[1] / ch
    f = lambda p: ((p[0] - x0) * sx, (p[1] - y0) * sy)
    e = {"inner": f(eye["inner"]), "outer": f(eye["outer"]), "root": [f(p) for p in eye["root"]]}
    if eye.get("lower"): e["lower"] = [f(p) for p in eye["lower"]]
    return e


def zone(eye, size=(OUT_W, OUT_H), up=0.7, down=0.08, lower=False, lower_depth=0.6):
    """속눈썹 존 (0~1, 페더). 윗 속눈썹: 뿌리 라인 위 up*w, 아래 down*w. lower=True면 눈 구멍 아래쪽 띠도 (약하게).
    eye에 lower 폴리라인이 있으면 눈 구멍(홍채·흰자)은 존에서 제외."""
    W, H = size
    inner, outer, root, w, axis, perp = eye_axes(eye)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    rx = np.array([p[0] for p in root]); ry = np.array([p[1] for p in root]); o = np.argsort(rx)
    lid_y = np.interp(xs[0], rx[o], ry[o])[None, :]
    d = (lid_y - ys) * (1 if perp[1] < 0 else -1) / w          # + = 속눈썹 쪽 (눈 너비 단위)
    t = ((xs - inner[0]) * axis[0] + (ys - inner[1]) * axis[1]) / w
    f = 0.05                                                   # 페더 폭
    sm = lambda x: np.clip(x / f, 0, 1)
    upper = sm(t + 0.3) * sm(1.4 - t) * sm(d + down) * sm(up - d)
    z = upper
    if lower:
        lo = sm(t + 0.2) * sm(1.3 - t) * sm(-d - down) * sm(d + lower_depth)
        if eye.get("lower"):                                    # 눈 구멍 제외: 아랫눈꺼풀 라인보다 위는 0
            lx = np.array([p[0] for p in eye["lower"]]); ly = np.array([p[1] for p in eye["lower"]]); lo_o = np.argsort(lx)
            low_y = np.interp(xs[0], lx[lo_o], ly[lo_o])[None, :]
            below = (ys - low_y) * (1 if perp[1] < 0 else -1) / w   # + = 아랫눈꺼풀보다 아래(얼굴 쪽)
            lo = lo * sm(below + 0.02)
        z = np.maximum(z, lo * 0.9)
    return z.astype(np.float32)


def align(a_in, a_out, zone_arr, max_shift=0.04, step=2):
    """출력을 입력에 평행이동으로 정렬 (존 밖 픽셀 MAE 최소). 1/4 축소로 탐색. returns (dx, dy) in full px."""
    H, W = zone_arr.shape
    s = 4
    li = Image.fromarray(_lum(a_in).astype(np.uint8)).resize((W // s, H // s)); lo = Image.fromarray(_lum(a_out).astype(np.uint8)).resize((W // s, H // s))
    li, lo = np.asarray(li, np.float32), np.asarray(lo, np.float32)
    zm = np.asarray(Image.fromarray((zone_arr * 255).astype(np.uint8)).resize((W // s, H // s)), np.float32) / 255
    wgt = (1 - zm)
    r = int(W * max_shift / s); best = (1e18, 0, 0)
    h, w = li.shape
    for dy in range(-r, r + 1, max(1, step // s)):
        for dx in range(-r, r + 1, max(1, step // s)):
            a = li[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)]
            b = lo[max(0, -dy):h + min(0, -dy), max(0, -dx):w + min(0, -dx)]
            m = wgt[max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)]
            err = float((np.abs(a - b) * m).sum() / (m.sum() + 1))
            if err < best[0]: best = (err, dx, dy)
    return best[1] * s, best[2] * s


def shift(img, dx, dy):
    """출력 이미지를 (dx,dy) 만큼 옮김: out[y+dy, x+dx] = in[y, x] 가 되도록 → 입력 좌표계에 맞춤."""
    return img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy), resample=Image.BILINEAR)


def darken_alpha(a_in, a_out, zone_arr, thr=10.0, soft=25.0):
    """입력보다 thr 이상 어두워진 곳만 (속눈썹). 존으로 제한, 1px 블러."""
    d = _lum(a_in) - _lum(a_out)
    a = np.clip((d - thr) / soft, 0, 1) * zone_arr
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)), np.float32) / 255
    return a


def composite_eye(frame, eye, out_img, box, lower=False, zone_kw=None):
    """frame(PIL) 에 한쪽 눈 결과를 얹는다. out_img = 모델 출력(크롭 프레이밍). returns (frame, debug dict)."""
    x0, y0, cw, ch = box
    crop_in = frame.crop((x0, y0, x0 + cw, y0 + ch)).resize((OUT_W, OUT_H), Image.LANCZOS)
    out = out_img.convert("RGB").resize((OUT_W, OUT_H), Image.LANCZOS)
    e = to_crop(eye, box)
    z = zone(e, lower=lower, **(zone_kw or {}))
    a_in, a_out = _np(crop_in), _np(out)
    dx, dy = align(a_in, a_out, z)
    out = shift(out, dx, dy); a_out = _np(out)
    alpha = darken_alpha(a_in, a_out, z)
    # 프레임 해상도에서 합성 (크롭 왕복 리샘플링으로 원본이 흐려지지 않게)
    a_small = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), np.float32)[..., None] / 255
    o_small = np.asarray(out.resize((cw, ch), Image.LANCZOS), np.float32)
    base = np.asarray(frame.convert("RGB"), np.float32)
    region = base[y0:y0 + ch, x0:x0 + cw]
    base[y0:y0 + ch, x0:x0 + cw] = region * (1 - a_small) + o_small * a_small
    res = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    dbg = {"crop_in": crop_in, "crop_out": out, "alpha": Image.fromarray((alpha * 255).astype(np.uint8)), "zone": Image.fromarray((z * 255).astype(np.uint8)),
           "shift": (dx, dy), "alpha_mean": float(alpha.mean()), "box": box}
    return res, dbg


def mask_png_b64(eye, box, lower=False):
    """gpt-image-1 edits 용 마스크: 존(>0.5)은 투명(편집 가능), 나머지 불투명. base64 PNG (1536x1024)."""
    import base64
    z = zone(to_crop(eye, box), lower=lower)
    a = ((z < 0.5) * 255).astype(np.uint8)
    m = Image.fromarray(np.dstack([np.zeros_like(a)] * 3 + [a]), "RGBA")
    buf = io.BytesIO(); m.save(buf, "PNG"); return base64.b64encode(buf.getvalue()).decode()


def crop_b64(frame, box, q=92):
    import base64
    x0, y0, cw, ch = box
    c = frame.crop((x0, y0, x0 + cw, y0 + ch)).resize((OUT_W, OUT_H), Image.LANCZOS)
    buf = io.BytesIO(); c.save(buf, "JPEG", quality=q); return base64.b64encode(buf.getvalue()).decode()


# ---------- 전이(transfer): 생성 결과에서 속눈썹만 뽑아 사용자 눈꺼풀 좌표계로 옮겨 붙임 ----------
# 눈꺼풀 좌표계 (u, v): u = 눈앞(0)→눈꼬리(1) 가로 비율, v = 뿌리 라인에서 속눈썹 쪽 거리 / 눈 너비.
# 생성 결과의 눈이 이동·변형돼도 (u,v) 로 뽑아서 사용자 (u,v) 에 그리면 실제 눈꺼풀에 정확히 붙는다.
from scipy.ndimage import map_coordinates


def _lid_fn(pts, perp_sign):
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts]); o = np.argsort(xs)
    return lambda x: np.interp(x, xs[o], ys[o]), perp_sign


def _uv_of(eye, which="root"):
    """(x,y) 배열 → (u,v). which: 'root'(윗 뿌리 라인) 또는 'lower'(아랫눈꺼풀). 반환 fn(x,y)->(u,v) 와 역함수."""
    inner, outer, root, w, axis, perp = eye_axes(eye)
    sign = 1 if perp[1] < 0 else -1                      # 화면 위쪽이 속눈썹 쪽이면 +1
    line = root if which == "root" else eye["lower"]
    f, _ = _lid_fn(line, sign)
    dxw = (outer[0] - inner[0]) or 1e-6
    def fwd(x, y):
        u = (x - inner[0]) / dxw
        v = (f(x) - y) * sign / w                      # 윗: +v = 위(속눈썹). 아랫: +v = 위 → 아래 속눈썹은 v<0
        return u, v
    def inv(u, v):
        x = inner[0] + u * dxw
        y = f(x) - v * sign * w
        return x, y
    return fwd, inv, w


def lash_map(out_img, eye_out, which="root", u_rng=(-0.35, 1.45), v_rng=(-0.04, 0.75), res=1000, thr=14.0, soft=34.0, gap=0.06):
    """생성 결과 → (u,v) 격자의 속눈썹 매트 (alpha, rgb). 배경(70퍼센타일)보다 어두운 픽셀. 눈썹은 갭 컷으로 제거."""
    a = _np(out_img); H, W = a.shape[:2]
    fwd, inv, w = _uv_of(eye_out, which)
    nu = res; nv = int(res * (v_rng[1] - v_rng[0]) / (u_rng[1] - u_rng[0]))
    us = np.linspace(u_rng[0], u_rng[1], nu); vs = np.linspace(v_rng[0], v_rng[1], nv)
    U, V = np.meshgrid(us, vs)
    X, Y = inv(U, V)
    lum = _lum(a)
    L = map_coordinates(lum, [Y, X], order=1, mode="nearest")
    rgb = np.stack([map_coordinates(a[..., c], [Y, X], order=0, mode="nearest") for c in range(3)], -1)   # 색은 최근접(속도)
    # 생성 결과 가장자리의 검은 띠(모델이 붙이는 패딩)는 제외: 평균 밝기 40 미만인 테두리 열/행을 잘라냄
    l = 0; r = W - 1; t = 0; b = H - 1
    while l < W // 4 and lum[:, l].mean() < 40: l += 1
    while r > W * 3 // 4 and lum[:, r].mean() < 40: r -= 1
    while t < H // 4 and lum[t].mean() < 40: t += 1
    while b > H * 3 // 4 and lum[b].mean() < 40: b -= 1
    inside = (X >= l + 3) & (X <= r - 3) & (Y >= t + 3) & (Y <= b - 3)
    bg = np.percentile(L[inside], 70) if inside.any() else 200
    alpha = np.clip((bg - L - thr) / soft, 0, 1) * inside
    # 붉은/갈색 픽셀(쌍꺼풀 선, 눈꺼풀 가장자리, 피부 그림자) 제외: 속눈썹은 무채색~짙은 갈색이고 아주 어둡다
    rb = rgb[..., 0] - rgb[..., 2]
    alpha = alpha * np.where(L > 40, 1 - np.clip((rb - 45) / 35, 0, 1), 1)   # 강한 붉은기만 (쌍꺼풀 선·점막). 가닥 가장자리의 갈색은 유지
    # 눈 바깥 구간(u<0 또는 u>1)은 눈썹·머리카락이 가까우니 높이 제한
    outside = (U < -0.02) | (U > 1.02)
    alpha = alpha * np.where(outside & (np.abs(V) > 0.45), 0, 1)
    # 눈썹/머리카락 컷: 각 u 열에서 뿌리에서 멀어지며 gap 이상 빈 구간이 나오면 그 위는 0 (열 단위 벡터화, 2026-10-04: 파이썬 이중 루프 3초 → 수십 ms)
    dark = alpha > 0.25; gap_n = max(1, int(gap * nv / (v_rng[1] - v_rng[0])))
    order = np.argsort(np.abs(vs))                       # 뿌리(v≈0)에서 먼 쪽으로
    nd = ~dark[order, :]                                 # 정렬된 행 × 열, True = 빈 칸
    far = np.abs(vs[order]) > 0.12
    run = np.zeros(nu, np.int32); cut_at = np.full(nu, nv, np.int32)   # 각 열의 컷 시작 행(정렬 기준)
    for r in range(nv):
        run = np.where(nd[r], run + 1, 0)
        hit = (run > gap_n) & far[r] & (cut_at == nv)
        cut_at[hit] = r
    rows = np.arange(nv)[:, None]
    kill = rows > cut_at[None, :]                        # 컷 행 이후(뿌리에서 더 먼 쪽) 전부 0
    a_sorted = alpha[order, :]; a_sorted[kill] = 0; alpha[order, :] = a_sorted
    return alpha.astype(np.float32), rgb.astype(np.float32), (us, vs)


def render_map(frame_crop, eye_in, amap, rgb, grid, which="root", blur=0.0, strength=1.0):
    """(u,v) 매트를 사용자 눈꺼풀 좌표계로 렌더해 frame_crop 위에 합성. returns (PIL, alpha_arr)."""
    a_in = _np(frame_crop); H, W = a_in.shape[:2]
    fwd, inv, w = _uv_of(eye_in, which)
    us, vs = grid
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    U, V = fwd(xs, ys)
    ui = (U - us[0]) / (us[-1] - us[0]) * (len(us) - 1); vi = (V - vs[0]) / (vs[-1] - vs[0]) * (len(vs) - 1)
    A = map_coordinates(amap, [vi, ui], order=1, mode="constant", cval=0.0) * strength
    RGB = np.stack([map_coordinates(rgb[..., c], [vi, ui], order=0, mode="nearest") for c in range(3)], -1)
    if blur > 0:
        A = np.asarray(Image.fromarray((np.clip(A, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur)), np.float32) / 255
    A = np.clip(A, 0, 1)[..., None]
    out = a_in * (1 - A) + RGB * A
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)), A[..., 0]


def lash_maps(out_img, eye_out, lower=False):
    """생성 결과 → (u,v) 매트들. u = 눈앞→눈꼬리 비율이라 반대쪽 눈에도 그대로 렌더할 수 있다."""
    out = out_img.convert("RGB").resize((OUT_W, OUT_H), Image.LANCZOS)
    maps = {"root": lash_map(out, eye_out, "root")}
    if lower and eye_out.get("lower"):
        maps["lower"] = lash_map(out, eye_out, "lower", u_rng=(0.06, 0.96), v_rng=(-0.45, -0.02))
    return maps


def render_eye(frame, eye, box, maps, blur=None):
    """(u,v) 매트들을 프레임의 한쪽 눈에 렌더. returns (frame, alpha_mean)."""
    x0, y0, cw, ch = box
    crop_in = frame.crop((x0, y0, x0 + cw, y0 + ch)).resize((OUT_W, OUT_H), Image.LANCZOS)
    e_in = to_crop(eye, box)
    if blur is None: blur = min(0.8, max(0.0, 0.6 * (OUT_W / cw - 1.0)))   # 폰 프레임은 선명해서 과한 블러가 가닥을 옅게 만듦 (2026-09-26 폰 2차: alpha 0.043)
    amap, rgb, grid = maps["root"]
    comp, A = render_map(crop_in, e_in, amap, rgb, grid, "root", blur=blur)
    if "lower" in maps and e_in.get("lower"):
        amap2, rgb2, grid2 = maps["lower"]
        comp, A2 = render_map(comp, e_in, amap2, rgb2, grid2, "lower", blur=blur)
        A = np.maximum(A, A2)
    comp_small = np.asarray(comp.resize((cw, ch), Image.LANCZOS), np.float32)
    a_small = np.asarray(Image.fromarray((A * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), np.float32)[..., None] / 255
    base = np.asarray(frame.convert("RGB"), np.float32)
    region = base[y0:y0 + ch, x0:x0 + cw]
    base[y0:y0 + ch, x0:x0 + cw] = region * (1 - a_small) + comp_small * a_small
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8)), float(A.mean())


def transfer_eye(frame, eye, out_img, eye_out, box, lower=False, blur=None):
    """crop → 매트 → 워프 → 프레임에 되붙임. eye/eye_out: 프레임 픽셀 / 출력(1536x1024) 픽셀 좌표.
    blur: None 이면 크롭 확대 배율로 자동 (원본이 흐린 만큼 속눈썹도 흐리게)."""
    x0, y0, cw, ch = box
    crop_in = frame.crop((x0, y0, x0 + cw, y0 + ch)).resize((OUT_W, OUT_H), Image.LANCZOS)
    out = out_img.convert("RGB").resize((OUT_W, OUT_H), Image.LANCZOS)
    e_in = to_crop(eye, box)
    if blur is None: blur = max(0.0, 0.6 * (OUT_W / cw - 1.0))
    amap, rgb, grid = lash_map(out, eye_out, "root")
    comp, A = render_map(crop_in, e_in, amap, rgb, grid, "root", blur=blur)
    alpha_total = A
    if lower and eye_out.get("lower") and e_in.get("lower"):
        amap2, rgb2, grid2 = lash_map(out, eye_out, "lower", u_rng=(0.06, 0.96), v_rng=(-0.45, -0.02))  # 눈구석·눈꼬리 점막은 제외
        comp, A2 = render_map(comp, e_in, amap2, rgb2, grid2, "lower", blur=blur)
        alpha_total = np.maximum(alpha_total, A2)
    # 프레임 해상도로 되붙임: 크롭 안에서 alpha 가 있는 곳만
    comp_small = np.asarray(comp.resize((cw, ch), Image.LANCZOS), np.float32)
    a_small = np.asarray(Image.fromarray((alpha_total * 255).astype(np.uint8)).resize((cw, ch), Image.LANCZOS), np.float32)[..., None] / 255
    base = np.asarray(frame.convert("RGB"), np.float32)
    region = base[y0:y0 + ch, x0:x0 + cw]
    base[y0:y0 + ch, x0:x0 + cw] = region * (1 - a_small) + comp_small * a_small
    res = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    dbg = {"crop_in": crop_in, "crop_out": out, "comp": comp, "alpha": Image.fromarray((alpha_total * 255).astype(np.uint8)),
           "lash_map": Image.fromarray((amap * 255).astype(np.uint8)), "blur": blur, "box": box}
    return res, dbg
