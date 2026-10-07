"""Fit the slate as a rigid rounded rectangle seen through a known camera.

SAM 3.1's mask covers only the *visible* board: a diver's hand on the edge
cuts a notch, and every frame has one. Labels follow the whole board (hidden
parts included), so a prediction that stops at the hand costs a manual drag
on nearly every frame.

A quadrilateral alone can't fix that: any convex quad is the image of *some*
rectangle. Two facts make it a real constraint:

- the frames are rectified with the camera's known intrinsics
  (``cameraintrinsics``), so projection is an exact pinhole;
- the board is a fixed rounded rectangle: one aspect ratio and one corner
  radius, measured from frames where the board is barely occluded.

So the fit is over the board's 3D pose (6 numbers), and its projected outline
must cover every pixel SAM saw (``MISS`` weight) while only lightly penalising
outline that SAM did not see (``EXTRA`` weight, since that's where hands are).
The visible edges then fix the hidden ones. Width is 1 (scale and distance
are the same unknown); height is ``aspect``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Every slate is the same blank board under different tape: on barely occluded frames
# of all four families (H, Tic-Tac-Toe, V, no template) the fitted aspect clusters at
# 0.70-0.72 and the corner radius at 3.1% of the width (2026-10-06, 59 frames).
# 0.708 is 8.5 x 12 in.
ASPECT = 0.708
RADIUS = 0.031
MISS = 1.0  # cost per SAM pixel left outside the outline
EXTRA = 0.15  # cost per outline pixel SAM did not see (hands, gear, shadow)
PHOTO_W = 4014


def scaled_K(camera_matrix, cache_w: int = 1600, photo_w: int = PHOTO_W) -> np.ndarray:
    """Intrinsics of the rectified photo, scaled to the frame cache."""
    K = np.array(camera_matrix, dtype=float).copy()
    K[:2] *= cache_w / photo_w
    return K


def board_outline(aspect: float, radius: float, n_arc: int = 6) -> np.ndarray:
    """(N, 3) points around a rounded rectangle, width 1 x height ``aspect``, on z = 0, clockwise in image."""
    r = float(np.clip(radius, 0.0, 0.49 * min(1.0, aspect)))
    if r <= 1e-6:
        pts = [(0, 0), (1, 0), (1, aspect), (0, aspect)]
    else:
        pts = []
        # corner centres and start angles, going TL -> TR -> BR -> BL (y down)
        for (cx, cy), a0 in (((r, r), np.pi), ((1 - r, r), 1.5 * np.pi), ((1 - r, aspect - r), 0.0), ((r, aspect - r), 0.5 * np.pi)):
            for t in np.linspace(a0, a0 + 0.5 * np.pi, n_arc):
                pts.append((cx + r * np.cos(t), cy + r * np.sin(t)))
    out = np.zeros((len(pts), 3))
    out[:, :2] = pts
    return out


def project(points3d: np.ndarray, rvec, tvec, K) -> np.ndarray:
    img, _ = cv2.projectPoints(points3d, np.asarray(rvec, float), np.asarray(tvec, float), K, None)
    return img.reshape(-1, 2)


def _clockwise(quad: np.ndarray) -> np.ndarray:
    """Order corners clockwise on screen (y down), starting at the top-left-most."""
    c = quad.mean(0)
    ang = np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0])
    q = quad[np.argsort(ang)]  # increasing angle = clockwise with y down
    start = int(np.argmin(q[:, 0] + q[:, 1]))
    return np.roll(q, -start, axis=0)


def pose_from_quad(quad, aspect: float, K, rotation: int | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """(rvec, tvec, reprojection rms px) mapping the board's corners onto ``quad``.

    ``rotation`` (0-3) forces which quad corner is the board's top-left; by default
    the best of the 4 is taken.
    """
    q = _clockwise(np.asarray(quad, float))
    obj = board_outline(aspect, 0.0)
    best = None
    for k in (range(4) if rotation is None else [rotation]):
        img = np.roll(q, -k, axis=0)
        ok, rv, tv = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if not ok or tv[2] <= 0:
            continue
        err = float(np.sqrt(((project(obj, rv, tv, K) - img) ** 2).sum(1).mean()))
        if best is None or err < best[2]:
            best = (rv.ravel(), tv.ravel(), err)
    if best is None:
        raise ValueError("no pose places the board in front of the camera")
    return best


def _line(p, q):
    """(a, b, c) with a x + b y = c."""
    a, b = q[1] - p[1], p[0] - q[0]
    return a, b, a * p[0] + b * p[1]


def _meet(l1, l2):
    (a1, b1, c1), (a2, b2, c2) = l1, l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    return np.array([(c1 * b2 - c2 * b1) / det, (a1 * c2 - a2 * c1) / det])


def min_enclosing_quad(mask: np.ndarray, eps: float = 1.5, max_edges: int = 14) -> np.ndarray | None:
    """Smallest quadrilateral, with sides along hull edges, that contains the mask.

    The starting guess for ``fit``: its sides lie on the board's visible straight
    edges. A side that is mostly hidden gets guessed from whatever hull edge is
    left, which is what the pose fit then corrects.
    """
    import itertools  # pylint: disable=import-outside-toplevel

    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    hull = cv2.convexHull(max(cs, key=cv2.contourArea)).astype(np.float32)
    poly = cv2.approxPolyDP(hull, eps, True).reshape(-1, 2).astype(float)
    n = len(poly)
    if n < 4:
        return None
    edges = [_line(poly[i], poly[(i + 1) % n]) for i in range(n)]
    lengths = [np.hypot(*(poly[(i + 1) % n] - poly[i])) for i in range(n)]
    longest = sorted(np.argsort(lengths)[::-1][:max_edges])
    best, best_area = None, np.inf
    for combo in itertools.combinations(longest, 4):
        corners = [_meet(edges[combo[k - 1]], edges[combo[k]]) for k in range(4)]
        if any(c is None for c in corners):
            continue
        q = np.array(corners)
        area = 0.5 * abs(np.dot(q[:, 0], np.roll(q[:, 1], -1)) - np.dot(q[:, 1], np.roll(q[:, 0], -1)))
        if not 0 < area < best_area:
            continue
        qc = q.astype(np.float32).reshape(-1, 1, 2)
        if cv2.contourArea(cv2.convexHull(qc)) < 0.999 * area:  # self-intersecting
            continue
        if all(cv2.pointPolygonTest(qc, (float(x), float(y)), True) >= -1.0 for x, y in poly):
            best, best_area = q, area
    return best


@dataclass
class RectFit:
    polygon: np.ndarray  # projected outline, frame-cache pixels
    corners: np.ndarray  # 4 sharp corners (TL, TR, BR, BL of the board), frame-cache pixels
    rvec: np.ndarray
    tvec: np.ndarray
    cost: float
    covered: float  # share of SAM's pixels inside the outline
    extra: float  # outline area SAM didn't see, as a share of the outline


def _raster(poly: np.ndarray, shape, offset) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.int32(np.round((poly - offset) * 4))], 1, shift=2)
    return m.astype(bool)


def tilt_deg(rvec) -> float:
    """Angle between the board's normal and the camera axis; 0 = facing the camera."""
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    return float(np.degrees(np.arccos(min(1.0, abs(R[2, 2])))))


def fit(mask: np.ndarray, K, aspect: float, radius: float, init_quad, pad: float = 0.35, rotation: int | None = None) -> RectFit:
    """Pose of the rounded board that best explains ``mask`` (see module docstring)."""
    from scipy.optimize import minimize  # pylint: disable=import-outside-toplevel

    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w, h = x1 - x0, y1 - y0
    X0, Y0 = max(0, int(x0 - pad * w)), max(0, int(y0 - pad * h))
    X1, Y1 = min(mask.shape[1], int(x1 + pad * w)), min(mask.shape[0], int(y1 + pad * h))
    crop = mask[Y0:Y1, X0:X1]
    offset = np.array([X0, Y0], float)
    n_mask = max(int(crop.sum()), 1)
    outline = board_outline(aspect, radius)
    rv0, tv0, _ = pose_from_quad(init_quad, aspect, K, rotation)

    def cost(x):
        rv, tv = x[:3], x[3:]
        if tv[2] <= 1e-3:
            return 1e9
        poly = project(outline, rv, tv, K)
        r = _raster(poly, crop.shape, offset)
        miss = np.count_nonzero(crop & ~r)
        extra = np.count_nonzero(r & ~crop)
        # the outline can run off the crop; count that as extra too
        inside = r.sum()
        area = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
        extra += max(0.0, area - inside)
        return (MISS * miss + EXTRA * extra) / n_mask

    x0v = np.concatenate([rv0, tv0])
    res = minimize(cost, x0v, method="Powell", options={"xtol": 1e-4, "ftol": 1e-5, "maxfev": 4000})
    rv, tv = res.x[:3], res.x[3:]
    poly = project(outline, rv, tv, K)
    r = _raster(poly, crop.shape, offset)
    return RectFit(
        polygon=poly, corners=project(board_outline(aspect, 0.0), rv, tv, K), rvec=rv, tvec=tv, cost=float(res.fun),
        covered=float((crop & r).sum() / n_mask), extra=float((r & ~crop).sum() / max(r.sum(), 1)),
    )


def start_quads(mask: np.ndarray) -> list[np.ndarray]:
    """Starting guesses for ``fit``. One guess can sit in a local minimum when a side is
    mostly hidden (the optimiser keeps its tilt), so ``fit_board`` tries all of them."""
    from slate_detector.maskqueue import fit_quad  # pylint: disable=import-outside-toplevel

    out = []
    q = min_enclosing_quad(mask)
    if q is not None:
        out.append(q)
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cs:
        out.append(cv2.boxPoints(cv2.minAreaRect(max(cs, key=cv2.contourArea))).astype(float))
    fq = fit_quad(mask)
    if fq is not None:
        out.append(np.asarray(fq, float))
    return out


# Divers hold the slate facing the camera, so a steep tilt needs strong evidence:
# fit_board picks the candidate minimising cost + TILT_PRIOR * (1 - cos tilt).
#
# Without it, two failure modes gave right outlines but wrong poses and depths:
# - a portrait-held board fitted as a landscape board tilted ~60 deg (cos 60 =
#   0.708^2: foreshortening the long side mimics the rotated outline);
# - a landscape board with hands over both side edges fitted at ~59 deg, its edge
#   slanted along the hand's outline so less of it counts as unseen (cost ~30%
#   lower than the upright pose; images 14789, 3719).
# On 442 test frames, 0.04 leaves 2 fits above 45 deg (from 145), cuts the worst
# frame's depth disagreement with sqrt(mask area) from 41% to 5.5%, and keeps box
# IoU with the original human slate rectangles at 0.955 (cscw-fishsense2027 e1j
# rectfit_sizes.py; 2026-10-07). A 30 deg tilt costs 0.13 * TILT_PRIOR.
TILT_PRIOR = 0.04


def fit_board(mask: np.ndarray, K, aspect: float = ASPECT, radius: float = RADIUS) -> RectFit | None:
    """Fit from every starting guess and every corner assignment (portrait and landscape
    readings both), then take the best under the tilt prior (``TILT_PRIOR``).

    None if no fit places the board in front of the camera.
    """
    fits = []
    for q in start_quads(mask):
        for rotation in range(4):
            try:
                fits.append(fit(mask, K, aspect, radius, q, rotation=rotation))
            except ValueError:
                continue
    if not fits:
        return None
    return min(fits, key=lambda f: f.cost + TILT_PRIOR * (1.0 - np.cos(np.radians(tilt_deg(f.rvec)))))


def plausible(f: RectFit, min_covered: float = 0.97, max_extra: float = 0.35) -> bool:
    """The outline holds what SAM saw, and isn't mostly unseen area (a wrong pose or a partial mask)."""
    return f.covered >= min_covered and f.extra <= max_extra


def clip_to_frame(poly: np.ndarray, width: int, height: int) -> np.ndarray:
    """Sutherland-Hodgman clip of a polygon to the image rectangle (boards cut by the frame edge)."""
    pts = [tuple(p) for p in np.asarray(poly, float)]
    for axis, bound, keep_ge in ((0, 0.0, True), (0, float(width), False), (1, 0.0, True), (1, float(height), False)):
        if not pts:
            break
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            cur_in = cur[axis] >= bound if keep_ge else cur[axis] <= bound
            prev_in = prev[axis] >= bound if keep_ge else prev[axis] <= bound
            if cur_in != prev_in:
                t = (bound - prev[axis]) / (cur[axis] - prev[axis])
                out.append((prev[0] + t * (cur[0] - prev[0]), prev[1] + t * (cur[1] - prev[1])))
            if cur_in:
                out.append(cur)
        pts = out
    return np.array(pts)


def fit_shape(mask: np.ndarray, K, init_quad, aspect0: float = 0.75, radius0: float = 0.03) -> tuple[float, float, RectFit]:
    """Fit pose *and* aspect/radius on a barely occluded board, to measure the board's shape."""
    from scipy.optimize import minimize  # pylint: disable=import-outside-toplevel

    def total(params):
        a, r = params
        if not 0.3 < a < 1.0 or not 0.0 <= r < 0.2:
            return 1e9
        try:
            return fit(mask, K, a, r, init_quad).cost
        except ValueError:
            return 1e9

    res = minimize(total, [aspect0, radius0], method="Nelder-Mead", options={"xatol": 2e-3, "fatol": 1e-4, "maxfev": 60})
    a, r = res.x
    return float(a), float(r), fit(mask, K, a, r, init_quad)
