from pathlib import Path
import math
import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "assets/video/demo-face-source-h264.mp4"
MODEL = ROOT / "assets/models/face_landmarker.task"
OUT_DIR = ROOT / "assets/video/tracked"
PREVIEW_DIR = ROOT / "assets/video/previews"

STYLES = {
    "natural": ROOT / "assets/lashes/airy-natural-v1.png",
    "cat": ROOT / "assets/lashes/soft-lift-v1.png",
    "wispy": ROOT / "assets/lashes/fairy-spike-v1.png",
    "volume": ROOT / "assets/lashes/velvet-volume-v1.png",
}

HEIGHT_FACTORS = {"natural": 0.20, "cat": 0.23, "wispy": 0.25, "volume": 0.27}
OPACITY_FACTORS = {"natural": 0.72, "cat": 0.76, "wispy": 0.70, "volume": 0.66}

# MediaPipe Face Mesh landmark indices: outer corner, inner corner, upper lid.
EYES = {
    "image_left": {
        "upper": (33, 246, 161, 160, 159, 158, 157, 173, 133),
        "lower": (33, 7, 163, 144, 145, 153, 154, 155, 133),
        "mirror": True,
    },
    "image_right": {
        "upper": (362, 398, 384, 385, 386, 387, 388, 466, 263),
        "lower": (362, 382, 381, 380, 374, 373, 390, 249, 263),
        "mirror": False,
    },
}


def alpha_composite(frame, overlay, x, y):
    fh, fw = frame.shape[:2]
    oh, ow = overlay.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + ow), min(fh, y + oh)
    if x1 >= x2 or y1 >= y2:
        return
    ox1, oy1 = x1 - x, y1 - y
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
    crop = overlay[oy1:oy2, ox1:ox2]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    rgb = crop[:, :, :3].astype(np.float32)
    dst = frame[y1:y2, x1:x2].astype(np.float32)
    frame[y1:y2, x1:x2] = (rgb * alpha + dst * (1.0 - alpha)).astype(np.uint8)


def transform_lash(asset, width, angle_deg, mirror):
    src = cv2.flip(asset, 1) if mirror else asset
    scale = width / src.shape[1]
    resized = cv2.resize(src, (max(2, int(src.shape[1] * scale)), max(2, int(src.shape[0] * scale))), interpolation=cv2.INTER_AREA)
    h, w = resized.shape[:2]
    center = (w / 2.0, h * 0.72)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    matrix[0, 2] += nw / 2.0 - center[0]
    matrix[1, 2] += nh / 2.0 - center[1]
    return cv2.warpAffine(resized, matrix, (nw, nh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def eye_curve(landmarks, spec, w, h):
    return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in spec["upper"]], dtype=np.float32)


def landmark_curve(landmarks, indices, w, h):
    return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices], dtype=np.float32)


def soften_asset(asset, opacity):
    """Make generated fibers less ink-black while preserving their fine detail."""
    result = asset.copy().astype(np.float32)
    warm_charcoal_bgr = np.array([38.0, 30.0, 35.0], dtype=np.float32)
    result[:, :, :3] = result[:, :, :3] * 0.62 + warm_charcoal_bgr * 0.38
    alpha = result[:, :, 3] / 255.0
    alpha = np.power(alpha, 0.92) * opacity
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.45)
    result[:, :, 3] = np.clip(alpha * 255.0, 0, 255)
    return result.astype(np.uint8)


def restore_eye_aperture(frame, original, upper, lower):
    """Put the real eye opening in front of the overlay for blink/eyelid occlusion."""
    polygon = np.vstack([upper, lower[::-1]])
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(np.round(polygon))], 255, lineType=cv2.LINE_AA)
    # Soft edge avoids a cut-out look while keeping fibers out of the eyeball.
    mask = cv2.GaussianBlur(mask, (0, 0), 0.75).astype(np.float32) / 255.0
    mask = mask[:, :, None]
    frame[:] = (original.astype(np.float32) * mask + frame.astype(np.float32) * (1.0 - mask)).astype(np.uint8)


def crop_alpha(asset):
    ys, xs = np.where(asset[:, :, 3] > 8)
    if not len(xs):
        raise RuntimeError("Empty lash alpha")
    pad = 2
    x1, x2 = max(0, xs.min() - pad), min(asset.shape[1], xs.max() + pad + 1)
    y1, y2 = max(0, ys.min() - pad), min(asset.shape[0], ys.max() + pad + 1)
    return asset[y1:y2, x1:x2]


def column_bounds(asset, x):
    radius = max(2, asset.shape[1] // 80)
    x1, x2 = max(0, x - radius), min(asset.shape[1], x + radius + 1)
    ys = np.where(asset[:, x1:x2, 3] > 12)[0]
    if not len(ys):
        return 0.0, float(asset.shape[0] - 1)
    return float(np.percentile(ys, 3)), float(np.percentile(ys, 96))


def warp_triangle(src, canvas, src_tri, dst_tri):
    src_tri = np.float32(src_tri)
    dst_tri = np.float32(dst_tri)
    sr = cv2.boundingRect(src_tri)
    dr = cv2.boundingRect(dst_tri)
    if sr[2] <= 1 or sr[3] <= 1 or dr[2] <= 1 or dr[3] <= 1:
        return
    sx, sy, sw, sh = sr
    dx, dy, dw, dh = dr
    src_crop = src[sy:sy + sh, sx:sx + sw]
    s_local = src_tri - np.array([sx, sy], dtype=np.float32)
    d_local = dst_tri - np.array([dx, dy], dtype=np.float32)
    matrix = cv2.getAffineTransform(s_local, d_local)
    warped = cv2.warpAffine(src_crop, matrix, (dw, dh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    mask = np.zeros((dh, dw), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(np.round(d_local)), 1.0, lineType=cv2.LINE_AA)
    warped[:, :, 3] = (warped[:, :, 3].astype(np.float32) * mask).astype(np.uint8)
    alpha_composite(canvas, warped, dx, dy)


def mesh_lash(frame, raw_asset, roots, mirror, height_factor):
    asset = crop_alpha(cv2.flip(raw_asset, 1) if mirror else raw_asset)
    n = len(roots)
    xs = np.linspace(0, asset.shape[1] - 1, n)
    src_top, src_root = [], []
    for x in xs:
        top, root = column_bounds(asset, int(round(x)))
        src_top.append((x, top))
        src_root.append((x, root))

    eye_width = float(np.linalg.norm(roots[-1] - roots[0]))
    lash_height = eye_width * height_factor
    dst_top = []
    for i, root in enumerate(roots):
        prev = roots[max(0, i - 1)]
        nxt = roots[min(n - 1, i + 1)]
        tangent = nxt - prev
        norm = max(1e-6, float(np.linalg.norm(tangent)))
        tangent /= norm
        normal = np.array([tangent[1], -tangent[0]], dtype=np.float32)
        if normal[1] > 0:
            normal *= -1
        # Keep the generated root curve seated a couple of pixels into the lid.
        roots[i] = root + np.array([0.0, eye_width * 0.018], dtype=np.float32)
        dst_top.append(roots[i] + normal * lash_height)
    dst_top = np.array(dst_top, dtype=np.float32)

    for i in range(n - 1):
        warp_triangle(asset, frame,
                      [src_top[i], src_root[i], src_top[i + 1]],
                      [dst_top[i], roots[i], dst_top[i + 1]])
        warp_triangle(asset, frame,
                      [src_root[i], src_root[i + 1], src_top[i + 1]],
                      [roots[i], roots[i + 1], dst_top[i + 1]])


def render_style(style, asset_path, landmarker):
    cap = cv2.VideoCapture(str(SOURCE))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = 720
    out_h = int(round(src_h * out_w / src_w / 2) * 2)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / f"{style}-tracked-raw.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
    asset = cv2.imread(str(asset_path), cv2.IMREAD_UNCHANGED)
    if asset is None or asset.shape[2] != 4:
        raise RuntimeError(f"Invalid alpha asset: {asset_path}")
    asset = soften_asset(asset, OPACITY_FACTORS[style])

    smooth = {}
    last_curve = {}
    last_lower = {}
    frame_no = 0
    detected = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        original = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_no * 1000.0 / fps)
        result = landmarker.detect_for_video(image, timestamp_ms)
        if result.face_landmarks:
            detected += 1
            landmarks = result.face_landmarks[0]
            for name, spec in EYES.items():
                curve = eye_curve(landmarks, spec, out_w, out_h)
                if name in smooth:
                    curve = smooth[name] * 0.68 + curve * 0.32
                smooth[name] = curve
                last_curve[name] = curve.copy()
                lower = landmark_curve(landmarks, spec["lower"], out_w, out_h)
                lower_key = f"{name}_lower"
                if lower_key in smooth:
                    lower = smooth[lower_key] * 0.68 + lower * 0.32
                smooth[lower_key] = lower
                last_lower[name] = lower.copy()

        for name, spec in EYES.items():
            if name not in last_curve:
                continue
            mesh_lash(frame, asset, last_curve[name].copy(), spec["mirror"], HEIGHT_FACTORS[style])

        # The natural eye opening is foreground. This handles blinks and prevents
        # the warped root/fibers from painting across the iris or lower eyelid.
        for name in EYES:
            if name in last_curve and name in last_lower:
                restore_eye_aperture(frame, original, last_curve[name], last_lower[name])

        writer.write(frame)
        if frame_no == max(1, total // 2):
            cv2.imwrite(str(PREVIEW_DIR / f"{style}-preview.jpg"), frame)
        frame_no += 1

    cap.release()
    writer.release()
    print(f"{style}: {frame_no} frames, face detected in {detected} frames, raw={raw_path}")


def main():
    base = mp.tasks.BaseOptions(model_asset_path=str(MODEL))
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        min_tracking_confidence=0.45,
    )
    for style, asset in STYLES.items():
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
            render_style(style, asset, landmarker)


if __name__ == "__main__":
    main()
