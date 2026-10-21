"""
SidewalkTrack - real-time multi-object detection, tracking and hazard logic
for an autonomous sidewalk delivery robot.

Run:   streamlit run app.py        (no API key needed; the live demo starts automatically)

Everything lives in this one file. Section banners mark the repo modules
(detect/, track/, reid/, geometry/, hazards/, io/, bench/) so it can be split later.
"""
import importlib.util
import io
import json
import math
import os
import tempfile
import time
import urllib.request
import zipfile
from collections import defaultdict, deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from scipy.optimize import linear_sum_assignment

try:
    import psutil
except Exception:  # memory profiling is optional
    psutil = None

CLASSES = ["person", "bicycle", "car", "dog", "stroller", "wheelchair", "other"]
CID = {c: i for i, c in enumerate(CLASSES)}
RISK_NAMES = ["NONE", "CAUTION", "STOP"]
RISK_BGR = {0: (90, 200, 90), 1: (0, 190, 255), 2: (50, 50, 240)}
BIG = 1e5
MAX_FRAMES = int(os.environ.get("SIDEWALKTRACK_MAX_FRAMES", "0"))  # 0 = run forever
MODES = {
    "bytetrack": "ByteTrack (baseline)",
    "reid_all": "ByteTrack + Re-ID (every frame)",
    "reid_gated": "ByteTrack + Re-ID (gated)",
}

# =====================================================================
#  io/ : YAML configuration (per-class thresholds, device profiles ...)
# =====================================================================
DEFAULT_YAML = """\
# ---- model selection: the same pipeline runs on laptop CPU / Jetson / Intel NUC ----
device_profiles:
  laptop_cpu: {format: onnx,     runtime: onnxruntime, imgsz: 640, half: false}
  jetson:     {format: tensorrt, runtime: tensorrt,    imgsz: 640, half: true}
  intel_nuc:  {format: openvino, runtime: openvino,    imgsz: 640, half: false}
detector:
  weights: yolov8n.pt        # or yolo11n.pt / custom sidewalk weights (adds stroller, wheelchair)
  nms_iou: 0.5
  class_conf: {person: 0.30, bicycle: 0.30, car: 0.35, dog: 0.35, stroller: 0.30, wheelchair: 0.30, other: 0.50}
tracker:
  high_thr: 0.50             # ByteTrack high-confidence pool
  low_thr: 0.10              # ByteTrack low-confidence pool
  new_track_thr: 0.60
  match_iou: 0.20
  low_match_iou: 0.50
  confirm_hits: 3            # tentative -> confirmed after 3 consecutive hits
  max_lost: 45               # frames before a lost track is removed
  reid_thr: 0.35             # max cosine distance for an appearance match
  emb_refresh: 15            # gated mode: refresh a track's embedding every N frames
  gate_px: 60                # spatial gate for re-ID candidates (at 960 px width)
hazards:
  robot_radius_m: 0.5
  zone_half_width_m: 1.2
  caution_len_m: 5.0
  stop_dist_m: 1.8
  stop_ttc_s: 2.0
  caution_ttc_s: 4.0
  coast_frames: 6
  hazard_classes: [person, bicycle, car, stroller, wheelchair]
  caution_only_classes: [dog]
calibration:                 # homography: image quad (normalised) <-> ground-plane quad (metres)
  world_quad: [[-3, 1], [3, 1], [3, 14], [-3, 14]]
  image_quad: [[0.10, 0.97], [0.90, 0.97], [0.60, 0.40], [0.40, 0.40]]
mqtt: {host: localhost, port: 1883, topic: sidewalktrack/hazards}
targets: {fps: 20, latency_ms: 150}
"""


def deep_merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_cfg(text):
    base = yaml.safe_load(DEFAULT_YAML)
    try:
        return deep_merge(base, yaml.safe_load(text) or {}), None
    except Exception as e:  # invalid YAML -> fall back to defaults
        return base, str(e)


class Publisher:
    """Hazard-event stream: in-memory JSON stream, plus MQTT when enabled."""

    def __init__(self):
        self.q = deque(maxlen=1000)
        self.client = None
        self.status = "JSON stream only"

    def connect(self, host, port):
        try:
            import paho.mqtt.client as mqtt
            ver = getattr(mqtt, "CallbackAPIVersion", None)
            self.client = mqtt.Client(ver.VERSION2) if ver else mqtt.Client()
            self.client.connect(host, int(port), 5)
            self.client.loop_start()
            self.status = f"MQTT connected ({host}:{port})"
        except Exception as e:
            self.client = None
            self.status = f"MQTT unavailable ({type(e).__name__}) - using JSON stream"

    def publish(self, topic, ev):
        self.q.append(ev)
        if self.client is not None:
            try:
                self.client.publish(topic, json.dumps(ev))
            except Exception:
                pass


# =====================================================================
#  geometry/ : calibrated homography, ground-plane mapping
# =====================================================================
class Geometry:
    def __init__(self, cfg, W, H):
        self.W, self.H = W, H
        wq = np.float32(cfg["calibration"]["world_quad"])
        iq = np.float32(cfg["calibration"]["image_quad"]) * np.float32([W, H])
        self.Hwi = cv2.getPerspectiveTransform(wq, iq)
        self.Hiw = np.linalg.inv(self.Hwi)

    def w2i(self, x, y):
        p = self.Hwi @ np.array([x, y, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def i2w(self, u, v):
        p = self.Hiw @ np.array([u, v, 1.0])
        return p[0] / p[2], p[1] / p[2]

    def box(self, x, y, wm, hm):
        """Image box (xyxy) of an upright object standing at ground point (x, y)."""
        u, v = self.w2i(x, y)
        ppm = abs(self.w2i(x + 0.5, y)[0] - self.w2i(x - 0.5, y)[0])
        return np.array([u - wm * ppm / 2, v - hm * ppm, u + wm * ppm / 2, v])

    def poly(self, half_w, y0, y1):
        pts = [self.w2i(-half_w, y0), self.w2i(half_w, y0), self.w2i(half_w, y1), self.w2i(-half_w, y1)]
        return np.round(np.array(pts)).astype(np.int32)


def iou_matrix(a, b):
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


# =====================================================================
#  simulated feed : procedurally generated sidewalk video (works instantly)
# =====================================================================
SPEC = {  # width m, height m, speed range m/s
    "person": (0.6, 1.75, (0.9, 1.7)), "bicycle": (0.8, 1.2, (2.5, 4.5)), "car": (2.0, 1.5, (3.0, 5.0)),
    "dog": (0.7, 0.55, (1.0, 2.5)), "stroller": (0.7, 1.1, (0.8, 1.3)), "wheelchair": (0.75, 1.3, (0.9, 1.8)),
}
SCEN = {
    "day": dict(miss=0.03, fp=0.15, dark=1.0, rain=False, max_obj=7, rate=0.55),
    "night": dict(miss=0.10, fp=0.30, dark=0.38, rain=False, max_obj=7, rate=0.55),
    "rain": dict(miss=0.07, fp=0.50, dark=0.80, rain=True, max_obj=7, rate=0.55),
    "crowded": dict(miss=0.05, fp=0.20, dark=1.0, rain=False, max_obj=16, rate=1.7),
}
MIX = {"person": .45, "bicycle": .12, "car": .05, "dog": .12, "stroller": .14, "wheelchair": .12}


def hsv2bgr(h, s=190, v=230):
    return tuple(int(x) for x in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


def draw_obj(img, kind, b, c1, c2):
    x1, y1, x2, y2 = [int(round(v)) for v in b]
    w, h = x2 - x1, y2 - y1
    if w < 3 or h < 3:
        return
    skin, dark = (150, 180, 230), (30, 30, 30)
    if kind == "person":
        hh = max(int(h * .18), 2)
        cv2.circle(img, (x1 + w // 2, y1 + hh // 2 + 1), max(hh // 2, 1), skin, -1)
        cv2.rectangle(img, (x1 + w // 8, y1 + hh), (x2 - w // 8, y1 + int(h * .58)), c1, -1)
        cv2.rectangle(img, (x1 + w // 5, y1 + int(h * .58)), (x2 - w // 5, y2), c2, -1)
    elif kind == "bicycle":
        r = max(h // 4, 2)
        for cx in (x1 + r, x2 - r):
            cv2.circle(img, (cx, y2 - r), r, dark, max(1, r // 5))
            cv2.line(img, (cx, y2 - r), (x1 + w // 2, y1 + int(h * .45)), c1, max(2, w // 25))
        cv2.circle(img, (x1 + w // 2, y1 + int(h * .1)), max(h // 12, 2), skin, -1)
        cv2.rectangle(img, (x1 + w // 2 - w // 6, y1 + int(h * .18)), (x1 + w // 2 + w // 6, y1 + int(h * .55)), c2, -1)
    elif kind == "car":
        cv2.rectangle(img, (x1, y1 + int(h * .35)), (x2, y2 - int(h * .18)), c1, -1)
        cv2.rectangle(img, (x1 + w // 5, y1), (x2 - w // 5, y1 + int(h * .4)), c1, -1)
        cv2.rectangle(img, (x1 + w // 4, y1 + int(h * .06)), (x2 - w // 4, y1 + int(h * .34)), (210, 215, 200), -1)
        for cx in (x1 + w // 5, x2 - w // 5):
            cv2.circle(img, (cx, y2 - int(h * .14)), max(int(h * .16), 2), (20, 20, 20), -1)
    elif kind == "dog":
        cv2.ellipse(img, (x1 + w // 2, y1 + int(h * .45)), (max(w // 2, 1), max(h // 4, 1)), 0, 0, 360, c1, -1)
        cv2.circle(img, (x2 - w // 8, y1 + int(h * .3)), max(h // 5, 2), c2, -1)
        for cx in (x1 + w // 5, x2 - w // 4):
            cv2.line(img, (cx, y1 + int(h * .6)), (cx, y2), c1, max(2, w // 18))
    elif kind == "stroller":
        cv2.rectangle(img, (x1, y1 + int(h * .3)), (x2 - w // 6, y2 - int(h * .2)), c1, -1)
        cv2.ellipse(img, (x1 + w // 3, y1 + int(h * .3)), (max(w // 3, 1), max(int(h * .3), 1)), 0, 180, 360, c2, -1)
        cv2.line(img, (x2 - w // 6, y1 + int(h * .3)), (x2, y1), dark, max(2, w // 25))
        for cx in (x1 + w // 5, x2 - w // 4):
            cv2.circle(img, (cx, y2 - int(h * .1)), max(int(h * .1), 2), dark, -1)
    else:  # wheelchair
        r = max(h // 3, 2)
        cv2.circle(img, (x1 + w // 2, y2 - r), r, dark, max(2, r // 6))
        cv2.rectangle(img, (x1 + w // 4, y1 + int(h * .45)), (x2 - w // 8, y2 - int(h * .35)), c1, -1)
        cv2.rectangle(img, (x1 + w // 4, y1 + int(h * .15)), (x1 + w // 2, y1 + int(h * .5)), c2, -1)
        cv2.circle(img, (x1 + int(w * .38), y1 + int(h * .08)), max(h // 14, 2), (150, 180, 230), -1)


class Sim:
    """Robot-frame world: +y forward, x lateral (metres). Objects move, the robot drives forward."""

    def __init__(self, geo, scenario="day", seed=7, fps=20, robot_speed=1.2):
        self.geo, self.name, self.sc = geo, scenario, SCEN[scenario]
        self.rng = np.random.default_rng(seed)
        self.fps, self.dt = fps, 1.0 / fps
        self.objs, self.bushes, self.nid = [], [], 1
        self.odo, self.base, self.speed, self.frame = 0.0, robot_speed, robot_speed, 0
        self.bg = self._make_bg()
        for _ in range(int(7 * fps)):  # warm start so the very first frame is populated
            self.step(self.base)
        self.frame = 0

    def _make_bg(self):
        g, W, H = self.geo, self.geo.W, self.geo.H
        vh = int(g.w2i(0, 1e5)[1])
        img = np.zeros((H, W, 3), np.uint8)
        img[:vh] = (235, 205, 160)
        rng, x = np.random.default_rng(3), 0
        while x < W:
            bw, bh = int(rng.integers(W // 14, W // 6)), int(rng.integers(int(vh * .35), max(vh, 40)))
            cv2.rectangle(img, (x, vh - bh), (x + bw, vh), tuple(int(c) for c in rng.integers(110, 170, 3)), -1)
            x += bw
        img[vh:] = (70, 130, 70)
        pts = np.round([g.w2i(-3.5, 0), g.w2i(3.5, 0), g.w2i(3.5, 200), g.w2i(-3.5, 200)]).astype(np.int32)
        cv2.fillPoly(img, [pts], (172, 172, 178))
        cv2.polylines(img, [pts], True, (110, 110, 115), 2)
        return img

    def spawn(self, kind=None, mode=None, x=None, y=None):
        r = self.rng
        kind = kind or r.choice(list(MIX), p=np.array(list(MIX.values())) / sum(MIX.values()))
        wm, hm, (s0, s1) = SPEC[kind]
        sp = r.uniform(s0, s1)
        mode = mode or ("cross" if kind == "car" else r.choice(["approach", "cross", "overtake"], p=[.4, .4, .2]))
        if mode == "approach":
            px, py, vx, vy = r.uniform(-2.6, 2.6), r.uniform(13, 16), r.normal(0, .1), -sp
        elif mode == "cross":
            side = r.choice([-1, 1])
            px, py, vx, vy = side * 4.3, r.uniform(3, 13), -side * sp, r.normal(0, .1)
        else:
            px, py, vx, vy = r.uniform(-2.5, 2.5), r.uniform(1.8, 4), 0.0, sp
        h1 = int(r.integers(0, 180))
        self.objs.append(dict(id=self.nid, cls=kind, x=px if x is None else x, y=py if y is None else y, vx=vx, vy=vy,
                              wm=wm, hm=hm, c1=hsv2bgr(h1), c2=hsv2bgr((h1 + int(r.integers(30, 120))) % 180, 150, 150)))
        self.nid += 1

    def step(self, robot_target):
        dt, r = self.dt, self.rng
        self.speed += float(np.clip(robot_target - self.speed, -2.5 * dt, 1.0 * dt))
        self.odo += self.speed * dt
        for o in self.objs:
            if o["cls"] in ("person", "dog", "stroller", "wheelchair") and r.random() < 0.012:
                f = r.uniform(0.3, 1.6)  # pedestrians change pace (breaks constant-velocity prediction)
                o["vx"] *= f
                o["vy"] *= f
            o["x"] += o["vx"] * dt
            o["y"] += (o["vy"] - self.speed) * dt
        for b in self.bushes:
            b["y"] -= self.speed * dt
        self.objs = [o for o in self.objs if 0.9 < o["y"] < 18 and abs(o["x"]) < 4.6]
        self.bushes = [b for b in self.bushes if b["y"] > 0.5]
        if len(self.objs) < self.sc["max_obj"] and r.random() < self.sc["rate"] * dt:
            self.spawn()
        if len(self.bushes) < 3 and r.random() < dt / 4.5:
            self.bushes.append(dict(x=r.choice([-1, 1]) * r.uniform(1.7, 3.2), y=16.0))
        self.frame += 1

    def render(self):
        g, W, H, r = self.geo, self.geo.W, self.geo.H, self.rng
        img = self.bg.copy()
        k0 = (-self.odo) % 2.0
        for i in range(24):
            y = k0 + 2.0 * i
            if y > 0.2:
                cv2.line(img, tuple(int(v) for v in g.w2i(-3.5, y)), tuple(int(v) for v in g.w2i(3.5, y)), (140, 140, 146), 1)
        items = [(o["y"], "o", o, g.box(o["x"], o["y"], o["wm"], o["hm"])) for o in self.objs]
        items += [(b["y"], "b", b, g.box(b["x"], b["y"], 1.5, 1.6)) for b in self.bushes]
        items.sort(key=lambda t: -t[0])
        bush_boxes = [(it[0], it[3]) for it in items if it[1] == "b"]
        gt = []
        for y, kind, o, b in items:
            if kind == "b":
                x1, y1, x2, y2 = b
                cv2.ellipse(img, (int((x1 + x2) / 2), int((y1 + y2) / 2)), (max(int((x2 - x1) / 2), 1), max(int((y2 - y1) / 2), 1)),
                            0, 0, 360, (55, 95, 55), -1)
                continue
            draw_obj(img, o["cls"], b, o["c1"], o["c2"])
            area = max((b[2] - b[0]) * (b[3] - b[1]), 1.0)
            occ = 0.0
            for by, bb in bush_boxes:
                if by < y:
                    occ += (max(0, min(b[2], bb[2]) - max(b[0], bb[0])) * max(0, min(b[3], bb[3]) - max(b[1], bb[1]))) / area
            vis = 1.0 - min(1.0, occ)
            cb = np.clip(b, [0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])
            if (cb[2] - cb[0]) * (cb[3] - cb[1]) < 0.4 * area:
                continue
            gt.append(dict(id=o["id"], cls=o["cls"], box=cb, vis=vis))
        # photometric effects
        sc = self.sc
        if sc["rain"]:
            ov = img.copy()
            for _ in range(140):
                x, yy = int(r.integers(0, W)), int(r.integers(0, H))
                cv2.line(ov, (x, yy), (x - 4, yy + 16), (225, 225, 225), 1)
            for _ in range(6):
                u, v = g.w2i(r.uniform(-2.8, 2.8), r.uniform(2, 10))
                cv2.ellipse(ov, (int(u), int(v)), (int(W * .05), int(H * .012)), 0, 0, 360, (235, 235, 240), -1)
            img = cv2.addWeighted(ov, 0.35, img, 0.65, 0)
        if sc["dark"] < 1:
            img = cv2.add(cv2.convertScaleAbs(img, alpha=sc["dark"]), (14, 5, 0, 0))
        return img, gt


def sim_detect(sim, gt):
    """Detector emulation: noisy ground truth with misses, jitter, low-confidence drop-outs, false positives."""
    r, sc, W, H = sim.rng, sim.sc, sim.geo.W, sim.geo.H
    boxes, scores, cls = [], [], []
    for o in gt:
        vis = o["vis"]
        if vis < 0.2 or r.random() < sc["miss"] + (1 - vis) * 0.9:
            continue
        conf = float(np.clip(r.normal(0.78 - 0.5 * (1 - vis), 0.12), 0.05, 0.99))
        if r.random() < 0.10:
            conf *= r.uniform(0.2, 0.6)
        b = o["box"].astype(float)
        w, h = b[2] - b[0], b[3] - b[1]
        b = b + r.normal(0, 0.025, 4) * np.array([w, h, w, h])
        boxes.append(b), scores.append(conf), cls.append(CID[o["cls"]])
    for _ in range(r.poisson(sc["fp"])):
        w, h = r.uniform(.03, .08) * W, r.uniform(.08, .2) * H
        x, y = r.uniform(.15, .85) * W, r.uniform(.4, .95) * H
        boxes.append(np.array([x, y - h, x + w, y])), scores.append(float(r.uniform(.1, .55))), cls.append(int(r.integers(0, 6)))
    if not boxes:
        return dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), cls=np.zeros(0, int))
    b = np.clip(np.array(boxes), 0, [W - 1, H - 1, W - 1, H - 1])
    keep = ((b[:, 2] - b[:, 0]) > 4) & ((b[:, 3] - b[:, 1]) > 4)
    return dict(boxes=b[keep], scores=np.array(scores)[keep], cls=np.array(cls, int)[keep])


# =====================================================================
#  detect/ : detectors, class-aware NMS, per-class thresholds
# =====================================================================
NAME_MAP = {"person": 0, "bicycle": 1, "motorcycle": 1, "car": 2, "truck": 2, "bus": 2, "dog": 3,
            "stroller": 4, "baby stroller": 4, "wheelchair": 5}


def nms(boxes, scores, thr):
    order, keep = scores.argsort()[::-1], []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        order = order[1:][iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0] <= thr]
    return np.array(keep, int)


def postprocess(d, dcfg):
    """Per-class confidence thresholds + class-aware NMS (both from YAML)."""
    if len(d["scores"]) == 0:
        return d
    thr = np.array([dcfg["class_conf"].get(CLASSES[c], 0.3) for c in d["cls"]])
    k = d["scores"] >= thr
    B, S, C = d["boxes"][k], d["scores"][k], d["cls"][k]
    keep = []
    for c in np.unique(C):
        idx = np.where(C == c)[0]
        keep += list(idx[nms(B[idx], S[idx], dcfg["nms_iou"])])
    keep = np.array(sorted(keep), int)
    return dict(boxes=B[keep], scores=S[keep], cls=C[keep])


class SimDetector:
    def __init__(self, sim):
        self.sim = sim

    def detect(self, img, ctx):
        return sim_detect(self.sim, ctx["gt"])


class MotionDetector:
    """Model-free fallback for uploaded video / webcam when YOLO is not installed."""

    def __init__(self):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=32, detectShadows=False)

    def detect(self, img, ctx=None):
        H, W = img.shape[:2]
        m = self.bg.apply(cv2.resize(img, None, fx=.5, fy=.5))
        m = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), np.ones((7, 7), np.uint8), iterations=2)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        B, S, C = [], [], []
        for c in cs:
            x, y, w, h = [v * 2 for v in cv2.boundingRect(c)]
            if w * h < 0.0008 * W * H:
                continue
            B.append([x, y, x + w, y + h]), S.append(0.6)
            C.append(0 if h > 1.3 * w else (2 if w > 1.8 * h and w > 0.15 * W else 1))
        if not B:
            return dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), cls=np.zeros(0, int))
        return dict(boxes=np.array(B, float), scores=np.array(S), cls=np.array(C, int))


class HogDetector:
    """Real-person detector built into OpenCV (HOG + linear SVM) - no model download needed."""

    def __init__(self):
        self.h = cv2.HOGDescriptor()
        self.h.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, img, ctx=None):
        s = min(1.0, 640.0 / img.shape[1])
        small = cv2.resize(img, None, fx=s, fy=s) if s < 1 else img
        rects, wts = self.h.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
        if len(rects) == 0:
            return dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), cls=np.zeros(0, int))
        r = np.array(rects, float) / s
        return dict(boxes=np.c_[r[:, 0], r[:, 1], r[:, 0] + r[:, 2], r[:, 1] + r[:, 3]],
                    scores=np.clip(0.4 + 0.3 * np.asarray(wts, float).ravel(), 0.1, 0.95), cls=np.zeros(len(rects), int))


YOLOX_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"  # Apache-2.0
YOLOX_MAP = {0: 0, 1: 1, 3: 1, 2: 2, 5: 2, 7: 2, 16: 3}  # COCO ids -> person, bicycle, (motorbike), car, (bus, truck), dog


def fetch_file(url, name, min_bytes):
    path = os.path.join(tempfile.gettempdir(), name)
    if os.path.exists(path) and os.path.getsize(path) > min_bytes:
        return path
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(path + ".part", "wb") as f:
            f.write(r.read())
        os.replace(path + ".part", path)
        return path
    except Exception:
        return None


class YoloxDetector:
    """YOLOX-nano (YOLO-family) ONNX model run through OpenCV DNN - real-person detection, auto-downloaded (3.6 MB)."""
    S = 416

    def __init__(self, onnx_path):
        self.net = cv2.dnn.readNetFromONNX(onnx_path)
        g, st = [], []
        for s in (8, 16, 32):
            n = self.S // s
            yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
            g.append(np.stack((xv, yv), 2).reshape(-1, 2))
            st.append(np.full((n * n, 1), s))
        self.grid, self.stride = np.concatenate(g).astype(np.float32), np.concatenate(st).astype(np.float32)

    def detect(self, img, ctx=None):
        h, w = img.shape[:2]
        k = min(self.S / h, self.S / w)
        nh, nw = int(h * k), int(w * k)
        can = np.full((self.S, self.S, 3), 114, np.uint8)
        can[:nh, :nw] = cv2.resize(img, (nw, nh))
        self.net.setInput(cv2.dnn.blobFromImage(can, 1.0, (self.S, self.S), swapRB=False))
        o = self.net.forward()[0]
        sc = o[:, 4] * o[:, 5:].max(1)
        cl = o[:, 5:].argmax(1)
        ok = (sc > 0.1) & np.isin(cl, list(YOLOX_MAP))
        if not ok.any():
            return dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), cls=np.zeros(0, int))
        o, sc, cl = o[ok], sc[ok], cl[ok]
        xy, wh = (o[:, :2] + self.grid[ok]) * self.stride[ok], np.exp(o[:, 2:4]) * self.stride[ok]
        b = np.c_[xy - wh / 2, xy + wh / 2] / k
        return dict(boxes=np.clip(b, 0, [w - 1, h - 1, w - 1, h - 1]), scores=sc.astype(float), cls=np.array([YOLOX_MAP[c] for c in cl], int))


DEMO_URL = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/vtest.avi"  # OpenCV sample video (Apache-2.0)


def fetch_demo_video():
    return fetch_file(DEMO_URL, "sidewalktrack_vtest.avi", 1_000_000)


@st.cache_resource(show_spinner="Loading YOLO model (first run exports ONNX / TensorRT / OpenVINO)...")
def load_yolo(weights, fmt, imgsz, half):
    from ultralytics import YOLO
    model = YOLO(weights)
    base = os.path.splitext(weights)[0]
    ext = {"onnx": ".onnx", "tensorrt": ".engine", "openvino": "_openvino_model"}[fmt]
    try:
        path = base + ext
        if not os.path.exists(path):
            path = model.export(format={"onnx": "onnx", "tensorrt": "engine", "openvino": "openvino"}[fmt],
                                imgsz=imgsz, half=bool(half and fmt == "tensorrt"))
        exported = YOLO(path, task="detect")
        exported.predict(np.zeros((64, 64, 3), np.uint8), imgsz=imgsz, verbose=False)  # smoke test the runtime
        return exported, f"{os.path.basename(str(path))} ({fmt})", model
    except Exception as e:  # broken onnxruntime / TensorRT / OpenVINO install -> use PyTorch weights
        return model, f"PyTorch weights ({fmt} backend failed: {type(e).__name__})", model


class YoloDetector:
    def __init__(self, model, imgsz, fallback=None):
        self.model, self.imgsz, self.fallback = model, imgsz, fallback

    def detect(self, img, ctx=None):
        try:
            r = self.model.predict(img, imgsz=self.imgsz, conf=0.05, iou=0.7, verbose=False)[0]
        except Exception:
            if self.fallback is None or self.model is self.fallback:
                raise
            self.model = self.fallback  # runtime failed mid-stream: switch to PyTorch weights
            r = self.model.predict(img, imgsz=self.imgsz, conf=0.05, iou=0.7, verbose=False)[0]
        B, S, C = [], [], []
        for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
            n = NAME_MAP.get(str(r.names[int(c)]).lower())
            if n is not None:
                B.append(b), S.append(float(s)), C.append(n)
        if not B:
            return dict(boxes=np.zeros((0, 4)), scores=np.zeros(0), cls=np.zeros(0, int))
        return dict(boxes=np.array(B, float), scores=np.array(S), cls=np.array(C, int))


def make_detector(o, cfg, sim, notes):
    d = o["det"]
    if d == "auto":
        d = "sim" if sim is not None else ("yolo" if importlib.util.find_spec("ultralytics") else "yolox")
    o = dict(o, det=d)
    if d == "yolox":
        p = fetch_file(YOLOX_URL, "yolox_nano.onnx", 1_000_000)
        try:
            det = YoloxDetector(p)
            notes.append("Detector: YOLOX-nano (ONNX via OpenCV DNN, COCO classes: person / bicycle / car / dog)")
            return det
        except Exception as e:
            notes.append(f"YOLOX-nano unavailable ({type(e).__name__}) - using OpenCV HOG people detector")
            return HogDetector()
    if d == "hog":
        return HogDetector()
    if o["det"] == "yolo":
        try:
            prof = cfg["device_profiles"][o["profile"]]
            model, desc, pt = load_yolo(cfg["detector"]["weights"], prof["format"], prof["imgsz"], prof["half"])
            notes.append(f"YOLO backend: {desc}")
            return YoloDetector(model, prof["imgsz"], pt)
        except Exception as e:
            notes.append(f"YOLO unavailable ({type(e).__name__}: install `ultralytics`) - using fallback detector")
            if sim is None:
                return make_detector(dict(o, det="yolox"), cfg, sim, notes)
    if o["det"] == "motion":
        return MotionDetector()
    return SimDetector(sim) if sim is not None else HogDetector()


# =====================================================================
#  reid/ : appearance embedding (OSNet-compatible interface, lightweight stand-in)
# =====================================================================
def embed(img, box):
    """Colour-layout embedding (upper/lower body hue histograms, Hellinger-normalised, unit length).
    Drop-in point for OSNet: return a unit vector for a box crop."""
    H, W = img.shape[:2]
    x1, y1, x2, y2 = box
    w = x2 - x1
    x1, x2, y1, y2 = int(max(0, x1 + .2 * w)), int(min(W, x2 - .2 * w)), int(max(0, y1)), int(min(H, y2))
    if x2 - x1 < 2 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(cv2.resize(img[y1:y2, x1:x2], (16, 32), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > 50) & (hsv[..., 2] > 40)).astype(np.uint8) * 255
    feats = []
    for sl in (slice(0, 16), slice(16, 32)):
        h = cv2.calcHist([np.ascontiguousarray(hsv[sl])], [0], np.ascontiguousarray(mask[sl]), [18], [0, 180]).ravel() + 1e-3
        h = np.convolve(np.r_[h[-1], h, h[0]], [.25, .5, .25], "valid")
        feats.append(h / h.sum())
    v = np.sqrt(np.concatenate(feats))
    return (v / np.linalg.norm(v)).astype(np.float32)


# =====================================================================
#  track/ : Kalman filter + ByteTrack (+ optional gated re-ID)
# =====================================================================
_W1, _V1 = 1 / 20, 1 / 160
_F = np.eye(8)
for _i in range(4):
    _F[_i, 4 + _i] = 1.0
_HM = np.eye(4, 8)


def kf_init(m):
    h = m[3]
    std = np.array([2 * _W1 * h, 2 * _W1 * h, 1e-2, 2 * _W1 * h, 10 * _V1 * h, 10 * _V1 * h, 1e-5, 10 * _V1 * h])
    return np.r_[m, np.zeros(4)], np.diag(std ** 2)


def kf_predict(mean, cov):
    h = mean[3]
    q = np.diag(np.square(np.r_[[_W1 * h, _W1 * h, 1e-2, _W1 * h], [_V1 * h, _V1 * h, 1e-5, _V1 * h]]))
    return _F @ mean, _F @ cov @ _F.T + q


def kf_update(mean, cov, z):
    h = mean[3]
    S = _HM @ cov @ _HM.T + np.diag(np.square([_W1 * h, _W1 * h, 1e-1, _W1 * h]))
    K = np.linalg.solve(S, _HM @ cov).T
    return mean + K @ (z - _HM @ mean), cov - K @ S @ K.T


def xyxy2xyah(b):
    w, h = b[2] - b[0], max(b[3] - b[1], 1.0)
    return np.array([b[0] + w / 2, b[1] + h / 2, w / h, h])


def lsa(cost, thr):
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    r, c = linear_sum_assignment(cost)
    mt = [(int(i), int(j)) for i, j in zip(r, c) if cost[i, j] <= thr]
    return mt, sorted(set(range(n)) - {i for i, _ in mt}), sorted(set(range(m)) - {j for _, j in mt})


class Track:
    def __init__(self, box, score, cls, f):
        self.mean, self.cov = kf_init(xyxy2xyah(box))
        self.state, self.id, self.hits, self.age, self.tsu = "tentative", -1, 1, 1, 0
        self.score, self.cls, self.emb, self.emb_f, self.wv, self.born = float(score), int(cls), None, -999, np.zeros(2), f

    @property
    def box(self):
        cx, cy, a, h = self.mean[:4]
        h = max(h, 1.0)
        w = max(a, 0.05) * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self):
        if self.state == "lost":
            self.mean = self.mean.copy()
            self.mean[7] = 0.0
        self.mean, self.cov = kf_predict(self.mean, self.cov)
        self.age += 1
        self.tsu += 1


class Tracker:
    """ByteTrack (two-stage high/low confidence association, Kalman motion) with optional appearance re-ID.
    modes: bytetrack | reid_all (embed every detection, every frame) | reid_gated (embed only motion-unmatched detections)"""

    def __init__(self, cfg, mode, geo, fps):
        self.c, self.mode, self.geo, self.fps = cfg, mode, geo, fps
        self.tracks, self.f, self._id = [], 0, 0
        self.n_emb, self.t_reid, self.recoveries = 0, 0.0, 0

    def _next_id(self):
        self._id += 1
        return self._id

    def _emb(self, img, box):
        t0 = time.perf_counter()
        e = embed(img, box)
        self.t_reid += time.perf_counter() - t0
        self.n_emb += 1
        return e

    def _gate(self, t, box):
        b = t.box
        d = math.hypot((box[0] + box[2]) / 2 - (b[0] + b[2]) / 2, (box[1] + box[3]) / 2 - (b[1] + b[3]) / 2)
        return d <= self.c["gate_px"] * (self.geo.W / 960) * (1 + 0.5 * min(t.tsu, 60))

    def _cost1(self, cand, hi, C, B, iou, emb):
        cost = 1 - iou
        for i, t in enumerate(cand):
            for k, j in enumerate(hi):
                if t.cls != C[j]:
                    cost[i, k] = BIG
                elif self.mode == "reid_all" and t.emb is not None and emb.get(j) is not None:
                    ed = 1 - float(t.emb @ emb[j])
                    if ed > self.c["reid_thr"] + 0.15 or (iou[i, k] <= 0 and not self._gate(t, B[j])):
                        cost[i, k] = BIG
                    else:
                        cost[i, k] = 0.5 * (1 - iou[i, k]) + 0.5 * ed
        return cost

    def _apply(self, t, j, B, S, C, img, emb, low=False, reid=False):
        z = xyxy2xyah(B[j])
        if reid and t.tsu > 3:
            t.mean, t.cov = kf_init(z)
        else:
            t.mean, t.cov = kf_update(t.mean, t.cov, z)
        t.tsu, t.score, t.cls = 0, float(S[j]), int(C[j])
        if t.state == "lost":
            t.state = "confirmed"
        elif t.state == "tentative":
            t.hits += 1
            if t.hits >= self.c["confirm_hits"]:
                t.state, t.id = "confirmed", self._next_id()
        else:
            t.hits += 1
        if self.mode != "bytetrack" and not low and t.state == "confirmed":
            if self.mode == "reid_all" and emb.get(j) is not None:
                t.emb, t.emb_f = emb[j], self.f
            elif self.mode == "reid_gated" and S[j] >= 0.6 and (t.emb is None or self.f - t.emb_f >= self.c["emb_refresh"]):
                e = emb[j] if j in emb else self._emb(img, B[j])
                if e is not None:
                    t.emb, t.emb_f = e, self.f

    def update(self, img, dets, dt=None):
        c = self.c
        self.f += 1
        self.n_emb, self.t_reid = 0, 0.0
        B, S, C = dets["boxes"], dets["scores"], dets["cls"]
        hi = np.where(S >= c["high_thr"])[0]
        lo = np.where((S >= c["low_thr"]) & (S < c["high_thr"]))[0]
        for t in self.tracks:
            t.predict()
        cand = [t for t in self.tracks if t.state in ("tentative", "confirmed")] + [t for t in self.tracks if t.state == "lost"]
        emb = {}
        if self.mode == "reid_all":
            for j in hi:
                emb[int(j)] = self._emb(img, B[j])
        matched, new = set(), []
        # ---- stage 1: all tracks vs high-confidence detections (IoU / fused IoU+appearance) ----
        iou = iou_matrix(np.array([t.box for t in cand]).reshape(-1, 4), B[hi])
        m1, ut, ud = lsa(self._cost1(cand, [int(j) for j in hi], C, B, iou, emb), 1 - c["match_iou"])
        for i, k in m1:
            self._apply(cand[i], int(hi[k]), B, S, C, img, emb)
            matched.add(id(cand[i]))
        # ---- stage 2: unmatched confirmed tracks vs low-confidence detections ----
        rem = [cand[i] for i in ut if cand[i].state == "confirmed"]
        unm2 = rem
        if rem and len(lo):
            cost2 = 1 - iou_matrix(np.array([t.box for t in rem]), B[lo])
            cost2[np.array([t.cls for t in rem])[:, None] != C[lo][None, :]] = BIG
            m2, ut2, _ = lsa(cost2, 1 - c["low_match_iou"])
            for i, k in m2:
                self._apply(rem[i], int(lo[k]), B, S, C, img, emb, low=True)
                matched.add(id(rem[i]))
            unm2 = [rem[i] for i in ut2]
        # ---- stage 3 (re-ID): only detections the motion model failed to associate get an embedding ----
        left = [int(hi[k]) for k in ud]
        pool = [t for t in [cand[i] for i in ut if cand[i].state == "lost"] + unm2 if t.emb is not None]
        if self.mode != "bytetrack" and pool and left:
            for j in left:
                if j not in emb:
                    emb[j] = self._emb(img, B[j])
            cst = np.full((len(pool), len(left)), BIG)
            for a, t in enumerate(pool):
                for b_, j in enumerate(left):
                    if t.cls == C[j] and emb[j] is not None and self._gate(t, B[j]):
                        d = 1 - float(t.emb @ emb[j])
                        if d <= c["reid_thr"]:
                            cst[a, b_] = d
            m3, _, ud3 = lsa(cst, c["reid_thr"])
            for a, b_ in m3:
                self._apply(pool[a], left[b_], B, S, C, img, emb, reid=True)
                matched.add(id(pool[a]))
                self.recoveries += 1
            left = [left[b_] for b_ in ud3]
        for j in left:
            if S[j] >= c["new_track_thr"]:
                t = Track(B[j], S[j], C[j], self.f)
                if emb.get(j) is not None:
                    t.emb, t.emb_f = emb[j], self.f
                new.append(t)
        for t in cand:
            if id(t) in matched:
                continue
            if t.state == "tentative":
                t.state = "removed"
            elif t.state == "confirmed":
                t.state = "lost"
            if t.state == "lost" and t.tsu > c["max_lost"]:
                t.state = "removed"
        self.tracks = [t for t in self.tracks if t.state != "removed"] + new
        return [self._describe(t) for t in self.tracks if t.state in ("confirmed", "lost") and t.id >= 0]

    def _describe(self, t):
        b = t.box
        cx, fy = (b[0] + b[2]) / 2, b[3]
        X, Y = self.geo.i2w(cx, fy)
        vx, vy = t.mean[4], t.mean[5] + t.mean[7] / 2
        X2, Y2 = self.geo.i2w(cx + vx, fy + vy)
        wv = np.array([X2 - X, Y2 - Y]) * self.fps
        t.wv = wv if t.age <= 4 else 0.6 * t.wv + 0.4 * wv
        return dict(id=t.id, cls=t.cls, box=b, state=t.state, tsu=t.tsu, age_s=t.age / self.fps,
                    ttl_s=max(0, self.c["max_lost"] - t.tsu) / self.fps, vel_px=(t.mean[4] * self.fps, t.mean[5] * self.fps),
                    world=(float(X), float(Y)), wv=(float(t.wv[0]), float(t.wv[1])), score=t.score)


# =====================================================================
#  hazards/ : safety zone, time-to-contact, risk levels, event stream
# =====================================================================
class HazardEngine:
    def __init__(self, cfg, topic="sidewalktrack/hazards"):
        self.c, self.cur, self.down, self.topic = cfg, {}, {}, topic

    def evaluate(self, tracks, frame, ts):
        c, per, events = self.c, {}, []
        for t in tracks:
            name = CLASSES[t["cls"]]
            if name not in c["hazard_classes"] and name not in c["caution_only_classes"]:
                continue
            if not (t["state"] == "confirmed" or (t["state"] == "lost" and t["tsu"] <= c["coast_frames"])):
                continue
            X, Y = t["world"]
            vx, vy = t["wv"]
            d = math.hypot(X, Y)
            closing = -(X * vx + Y * vy) / max(d, 1e-3)
            ttc = (d - c["robot_radius_m"]) / closing if closing > 0.05 else math.inf
            in_zone = abs(X) <= c["zone_half_width_m"] and 0 <= Y <= c["caution_len_m"]
            tt = min(ttc, 3.0) if math.isfinite(ttc) else 0.0
            enters = ttc < c["caution_ttc_s"] and abs(X + vx * tt) <= c["zone_half_width_m"] and (Y + vy * tt) <= c["caution_len_m"]
            raw = 0
            if in_zone or enters:
                raw = 1
                if in_zone and (ttc <= c["stop_ttc_s"] or d <= c["stop_dist_m"]):
                    raw = 2
            if name in c["caution_only_classes"]:
                raw = min(raw, 1)
            i = t["id"]
            cur = self.cur.get(i, 0)
            if raw > cur:
                new, self.down[i] = raw, 0
            elif raw < cur:  # hysteresis: de-escalate after 5 consecutive lower readings
                self.down[i] = self.down.get(i, 0) + 1
                new = raw if self.down[i] >= 5 else cur
            else:
                new, self.down[i] = cur, 0
            if new != cur:
                events.append(dict(ts=round(time.time(), 3), iso=time.strftime("%H:%M:%S"), frame=frame, track_id=i, cls=name,
                                   risk=RISK_NAMES[new], ttc_s=round(ttc, 2) if math.isfinite(ttc) else None,
                                   dist_m=round(d, 2), pos_m=[round(X, 2), round(Y, 2)]))
            self.cur[i] = new
            per[i] = (new, ttc, d)
        live = {t["id"] for t in tracks}
        self.cur = {k: v for k, v in self.cur.items() if k in live}
        return per, max([v[0] for v in per.values()], default=0), events


# =====================================================================
#  bench/ : MOT metrics (MOTA, IDF1, ID switches), TrackEval export, ablation
# =====================================================================
class MOTEval:
    """CLEAR-MOT / IDF1 (IoU>=0.5). Approximation for quick ablations - export the MOT files and use TrackEval for HOTA."""

    def __init__(self):
        self.gt_n = self.fn = self.fp = self.idsw = 0
        self.last, self.pair, self.gtc, self.prc, self.events = {}, defaultdict(int), defaultdict(int), defaultdict(int), []

    def add(self, f, gt, pred, tag=None):
        ng, npd = len(gt), len(pred)
        self.gt_n += ng
        for g, _ in gt:
            self.gtc[g] += 1
        for p, _ in pred:
            self.prc[p] += 1
        mt = []
        if ng and npd:
            iou = iou_matrix([b for _, b in gt], [b for _, b in pred])
            bonus = np.array([[1.0 if self.last.get(g) == p else 0.0 for p, _ in pred] for g, _ in gt])
            score = np.where(iou >= 0.5, iou + bonus, 0)
            r, c = linear_sum_assignment(-score)
            mt = [(i, j) for i, j in zip(r, c) if score[i, j] > 0]
        self.fn += ng - len(mt)
        self.fp += npd - len(mt)
        for i, j in mt:
            g, p = gt[i][0], pred[j][0]
            self.pair[(g, p)] += 1
            if g in self.last and self.last[g] != p:
                self.idsw += 1
                self.events.append(dict(tag=tag, frame=f, gt_id=str(g), old_pred_id=str(self.last[g]), new_pred_id=str(p)))
            self.last[g] = p

    def result(self):
        mota = 1 - (self.fn + self.fp + self.idsw) / max(self.gt_n, 1)
        gids, pids = sorted(self.gtc, key=str), sorted(self.prc, key=str)
        idtp = 0
        if gids and pids:
            gi, pi = {g: i for i, g in enumerate(gids)}, {p: i for i, p in enumerate(pids)}
            M = np.zeros((len(gids), len(pids)))
            for (g, p), n in self.pair.items():
                M[gi[g], pi[p]] = n
            r, c = linear_sum_assignment(-M)
            idtp = int(M[r, c].sum())
        tg, tp = sum(self.gtc.values()), sum(self.prc.values())
        idf1 = 2 * idtp / max(2 * idtp + (tg - idtp) + (tp - idtp), 1)
        return dict(MOTA=mota, IDF1=idf1, IDSW=self.idsw, FN=self.fn, FP=self.fp, GT=self.gt_n)


def trackeval_zip(gt_rows, pred_rows, W, H, fps):
    buf = io.BytesIO()
    names = sorted(gt_rows)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("data/gt/mot_challenge/seqmaps/SIDEWALK-train.txt", "name\n" + "\n".join(names) + "\n")
        for n in names:
            z.writestr(f"data/gt/mot_challenge/SIDEWALK-train/{n}/gt/gt.txt", "\n".join(gt_rows[n]) + "\n")
            n_fr = max(int(r.split(",")[0]) for r in gt_rows[n]) if gt_rows[n] else 1
            z.writestr(f"data/gt/mot_challenge/SIDEWALK-train/{n}/seqinfo.ini",
                       f"[Sequence]\nname={n}\nimDir=img1\nframeRate={fps}\nseqLength={n_fr}\nimWidth={W}\nimHeight={H}\nimExt=.jpg\n")
            for m, rows in pred_rows.items():
                z.writestr(f"data/trackers/mot_challenge/SIDEWALK-train/SidewalkTrack-{m}/data/{n}.txt", "\n".join(rows.get(n, [])) + "\n")
    return buf.getvalue()


def run_ablation(cfg, seeds, frames, fps, progress):
    W, H = 640, 360
    geo, scns, modes = Geometry(cfg, W, H), list(SCEN), list(MODES)
    ev = {(m, s): MOTEval() for m in modes for s in scns + ["ALL"]}
    tm = {m: dict(t=0.0, n=0, emb=0, rec=0) for m in modes}
    gt_rows, pred_rows = {}, {m: {} for m in modes}
    total, done, clip = len(scns) * seeds * frames, 0, 0
    for scn in scns:
        for sd in range(seeds):
            clip += 1
            name = f"{scn}_{sd + 1:02d}"
            sim = Sim(geo, scn, seed=100 * sd + sum(map(ord, scn)) % 97, fps=fps)
            trs = {m: Tracker(cfg["tracker"], m, geo, fps) for m in modes}
            gt_rows[name] = []
            for m in modes:
                pred_rows[m][name] = []
            for f in range(1, frames + 1):
                sim.step(sim.base)
                img, gt = sim.render()
                gtl = [((clip, o["id"]), o["box"]) for o in gt if o["vis"] >= 0.4]
                dets = postprocess(sim_detect(sim, gt), cfg["detector"])
                gt_rows[name] += [f"{f},{o['id']},{o['box'][0]:.1f},{o['box'][1]:.1f},{o['box'][2] - o['box'][0]:.1f},"
                                  f"{o['box'][3] - o['box'][1]:.1f},1,1,1" for o in gt if o["vis"] >= 0.4]
                for m in modes:
                    t0 = time.perf_counter()
                    out = trs[m].update(img, dets)
                    tm[m]["t"] += time.perf_counter() - t0
                    tm[m]["n"] += 1
                    tm[m]["emb"] += trs[m].n_emb
                    pl = [((clip, o["id"]), o["box"]) for o in out if o["state"] == "confirmed" and o["tsu"] == 0]
                    ev[(m, scn)].add(f, gtl, pl, tag=name)
                    ev[(m, "ALL")].add(f, gtl, pl, tag=name)
                    pred_rows[m][name] += [f"{f},{k[1]},{b[0]:.1f},{b[1]:.1f},{b[2] - b[0]:.1f},{b[3] - b[1]:.1f},1,-1,-1,-1" for k, b in pl]
                done += 1
                if done % 20 == 0:
                    progress.progress(min(done / total, 1.0), text=f"Clip {clip}/{len(scns) * seeds}: {name}")
            for m in modes:
                tm[m]["rec"] += trs[m].recoveries
    progress.progress(1.0, text="Done")
    rows = []
    for m in modes:
        r = ev[(m, "ALL")].result()
        ms = 1000 * tm[m]["t"] / max(tm[m]["n"], 1)
        rows.append({"Tracker": MODES[m], "MOTA": round(r["MOTA"], 3), "IDF1": round(r["IDF1"], 3), "ID switches": r["IDSW"],
                     "Embeds/frame": round(tm[m]["emb"] / max(tm[m]["n"], 1), 2), "Re-ID recoveries": tm[m]["rec"],
                     "Tracker ms/frame": round(ms, 2), "Tracker FPS": round(1000 / max(ms, 1e-6), 0)})
    per = []
    for s in scns:
        for m in modes:
            r = ev[(m, s)].result()
            per.append({"Scenario": s, "Tracker": MODES[m], "MOTA": round(r["MOTA"], 3), "IDF1": round(r["IDF1"], 3), "ID switches": r["IDSW"]})
    fails = [dict(mode=MODES[m], **e) for m in modes for e in ev[(m, "ALL")].events][:60]
    return dict(summary=pd.DataFrame(rows), per=pd.DataFrame(per), fails=pd.DataFrame(fails), clips=clip,
                zip=trackeval_zip(gt_rows, pred_rows, W, H, fps), frames=frames)


# =====================================================================
#  io/ : pipeline world (source -> detect -> track -> re-ID -> hazards -> overlay)
# =====================================================================
def draw_overlay(img, dets, out, geo, hz, per, overall, o, speed):
    vis, W, H = img.copy(), geo.W, geo.H
    sc = W / 960
    if o["blur"]:  # privacy: blur head region of every detected person before anything is drawn/recorded
        for b, s, c in zip(dets["boxes"], dets["scores"], dets["cls"]):
            if c == 0 and s >= 0.3:
                x1, y1, x2, y2 = [int(v) for v in b]
                x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y1 + int(0.22 * (y2 - y1)) + 1, H)
                if x2 > x1 and y2 > y1:
                    k = max(3, (x2 - x1) // 3) | 1
                    vis[y1:y2, x1:x2] = cv2.GaussianBlur(vis[y1:y2, x1:x2], (k, k), 0)
    hw = hz["zone_half_width_m"]
    p_caut, p_stop = geo.poly(hw, 0.0, hz["caution_len_m"]), geo.poly(hw, 0.0, hz["stop_dist_m"])
    ov = vis.copy()
    cv2.fillPoly(ov, [p_caut], (0, 200, 255))
    cv2.fillPoly(ov, [p_stop], (50, 50, 240))
    vis = cv2.addWeighted(ov, 0.22, vis, 0.78, 0)
    cv2.polylines(vis, [p_caut], True, (0, 200, 255), 2)
    cv2.polylines(vis, [p_stop], True, (50, 50, 240), 2)
    for t in out:
        if t["state"] == "lost" and o["show_lost"] and t["tsu"] <= 20:
            x1, y1, x2, y2 = [int(v) for v in t["box"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (150, 150, 150), 1)
            cv2.putText(vis, f"#{t['id']} lost", (x1, max(y1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * sc + .1, (170, 170, 170), 1)
        if t["state"] != "confirmed" or t["tsu"] != 0:
            continue
        risk, ttc, dist = per.get(t["id"], (0, math.inf, 0))
        col = RISK_BGR[risk]
        x1, y1, x2, y2 = [int(v) for v in t["box"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2 if risk < 2 else 3)
        label = f"#{t['id']} {CLASSES[t['cls']]}" + (f" {RISK_NAMES[risk]} TTC {ttc:.1f}s" if risk and math.isfinite(ttc) else "")
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc + .1, 1)
        cv2.rectangle(vis, (x1, max(y1 - th - 6, 0)), (x1 + tw + 4, max(y1, th + 6)), col, -1)
        cv2.putText(vis, label, (x1 + 2, max(y1 - 4, th + 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * sc + .1, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(vis, (0, 0), (int(W * .33), int(34 * sc + 6)), RISK_BGR[overall], -1)
    cv2.putText(vis, f"ROBOT: {['CLEAR', 'CAUTION', 'STOP'][overall]}  {speed:.1f} m/s", (10, int(24 * sc + 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * sc + .1, (0, 0, 0), 2, cv2.LINE_AA)
    return vis


class World:
    def __init__(self, cfg, o, pub=None, seed=7):
        self.cfg, self.o, self.pub = cfg, dict(o), pub
        self.W, self.H, self.fps = o["W"], o["H"], o["fps"]
        self.geo = Geometry(cfg, self.W, self.H)
        self.notes, self.sim, self.cap = [], None, None
        if o["source"] != "sim":
            self.cap = cv2.VideoCapture(o["video_path"] if o["source"] == "video" else 0)
            if not self.cap.isOpened():
                self.notes.append("Could not open the video source - showing the simulated feed instead.")
                self.cap, self.o["source"] = None, "sim"
        if self.cap is None:
            self.sim = Sim(self.geo, o["scenario"], seed, self.fps)
        self.det = make_detector(self.o, cfg, self.sim, self.notes)
        self.trk = Tracker(cfg["tracker"], o["mode"], self.geo, self.fps)
        self.hz = HazardEngine(cfg["hazards"])
        self.frame, self.risk, self.speed = 0, 0, 1.2
        self.stage = defaultdict(lambda: deque(maxlen=40))
        self.events = deque(maxlen=200)
        self.nemb = deque(maxlen=40)

    def close(self):
        if self.cap is not None:
            self.cap.release()

    def _grab(self):
        ok, img = self.cap.read()
        if not ok and self.o["source"] == "video":
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self.cap.read()
        canvas = np.zeros((self.H, self.W, 3), np.uint8)
        if ok:
            h, w = img.shape[:2]
            k = min(self.W / w, self.H / h)
            nw, nh = int(w * k), int(h * k)
            y0, x0 = (self.H - nh) // 2, (self.W - nw) // 2
            canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh))
        return canvas

    def spawn_hazard(self):
        if self.sim is not None:
            self.sim.spawn("person", "approach", x=float(self.sim.rng.uniform(-.4, .4)), y=8.0)

    def step(self):
        pc, ts = time.perf_counter, {}
        t0 = pc()
        gt = []
        if self.sim is not None:
            tgt = self.sim.base
            if self.o["obey"]:
                tgt = self.sim.base * (0.0 if self.risk == 2 else 0.55 if self.risk == 1 else 1.0)
            self.sim.step(tgt)
            img, gt = self.sim.render()
            self.speed = self.sim.speed
        else:
            img = self._grab()
        ts["capture/sim"] = pc() - t0
        t0 = pc()
        dets = postprocess(self.det.detect(img, {"gt": gt}), self.cfg["detector"])
        ts["detect"] = pc() - t0
        t0 = pc()
        out = self.trk.update(img, dets)
        el = pc() - t0
        ts["reid"] = self.trk.t_reid
        ts["track"] = max(el - self.trk.t_reid, 0.0)
        t0 = pc()
        per, overall, evs = self.hz.evaluate(out, self.frame, time.time())
        ts["hazards"] = pc() - t0
        self.risk = overall
        topic = self.cfg["mqtt"]["topic"]
        for e in evs:
            self.events.appendleft(e)
            if self.pub is not None:
                self.pub.publish(topic, e)
        t0 = pc()
        vis = draw_overlay(img, dets, out, self.geo, self.cfg["hazards"], per, overall, self.o, self.speed)
        ts["overlay"] = pc() - t0
        self.frame += 1
        for k, v in ts.items():
            self.stage[k].append(v * 1000)
        self.nemb.append(self.trk.n_emb)
        lat = sum(ts[k] for k in ("detect", "track", "reid", "hazards")) * 1000
        rows = []
        for t in out:
            if t["state"] == "confirmed" and t["tsu"] == 0:
                r, ttc, d = per.get(t["id"], (0, math.inf, 0))
                rows.append({"ID": t["id"], "Class": CLASSES[t["cls"]], "Risk": RISK_NAMES[r], "Dist m": round(t["world"][1], 1),
                             "TTC s": round(ttc, 1) if math.isfinite(ttc) else None,
                             "Speed m/s": round(math.hypot(*t["wv"]), 1), "Age s": round(t["age_s"], 1), "Time-to-lost s": round(t["ttl_s"], 1)})
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), dict(latency=lat, rows=rows, overall=overall, n_tracks=len(rows))

    def profile(self):
        m = {k: float(np.mean(v)) for k, v in self.stage.items()}
        pipe = sum(m.get(k, 0) for k in ("detect", "track", "reid", "hazards"))
        return m, pipe


def rss_mb():
    return psutil.Process().memory_info().rss / 1e6 if psutil else float("nan")


def open_writer(path, fps, W, H):
    try:
        import imageio
        return "iio", imageio.get_writer(path, fps=fps, codec="libx264", macro_block_size=1, pixelformat="yuv420p")
    except Exception:
        return "cv", cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))


def record_clip(cfg, o, seconds=10):
    o2 = dict(o) if o["source"] == "video" else dict(o, source="sim", det="sim")
    w = World(cfg, o2, pub=None, seed=int(time.time()) % 1000)
    path = os.path.join(tempfile.mkdtemp(), "sidewalktrack_demo.mp4")
    kind, wr = open_writer(path, o["fps"], o["W"], o["H"])
    for i in range(int(seconds * o["fps"])):
        if i == int(2.5 * o["fps"]):
            w.spawn_hazard()
        vis, _ = w.step()
        wr.append_data(vis) if kind == "iio" else wr.write(cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    wr.close() if kind == "iio" else wr.release()
    with open(path, "rb") as f:
        return f.read()


def bench_detector(cfg, o, n=40):
    w = World(cfg, dict(o) if o["source"] == "video" else dict(o, source="sim"), pub=None)
    ms = []
    for _ in range(n):
        if w.sim is not None:
            w.sim.step(w.sim.base)
            img, gt = w.sim.render()
        else:
            img, gt = w._grab(), []
        t0 = time.perf_counter()
        w.det.detect(img, {"gt": gt})
        ms.append((time.perf_counter() - t0) * 1000)
    return ms, w.notes, type(w.det).__name__


# =====================================================================
#  Streamlit UI
# =====================================================================
_IMG_MODE = {"m": None}


def show_image(ph, frame):
    for kw in ({"width": "stretch"}, {"use_container_width": True}, {}):
        if _IMG_MODE["m"] is not None and _IMG_MODE["m"] != kw.__repr__():
            continue
        try:
            ph.image(frame, channels="RGB", output_format="JPEG", **kw)
            _IMG_MODE["m"] = kw.__repr__()
            return
        except Exception:
            _IMG_MODE["m"] = None
    ph.image(frame, channels="RGB", output_format="JPEG")


def card_grid(items):
    html = "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px'>"
    for label, val, ok in items:
        edge = {True: "#2ecc71", False: "#e74c3c", None: "rgba(128,128,128,.4)"}[ok]
        html += (f"<div style='border:1px solid {edge};border-left:5px solid {edge};border-radius:8px;padding:6px 10px;"
                 f"background:rgba(128,128,128,.10)'><div style='font-size:.72rem;opacity:.75'>{label}</div>"
                 f"<div style='font-size:1.25rem;font-weight:700'>{val}</div></div>")
    return html + "</div>"


ABOUT = """
### What this app is
A complete, self-contained **detect → track → re-ID → hazard** pipeline for a sidewalk delivery robot, running on a **procedurally generated sidewalk feed** so it works instantly with no downloads or API keys.
The default source is a **real pedestrian video** (OpenCV's public sample, downloaded once) with real people detected by **YOLOX-nano** (a YOLO-family ONNX model, auto-downloaded, 3.6 MB; falls back to OpenCV HOG). Choose *Simulated sidewalk feed* for the night / rain / crowded scenarios, or *Upload video* / *Webcam* for your own footage (install `ultralytics` to use real YOLO weights).

### Module map (single file, split-ready)
| Repo folder | Section in `app.py` | Notes |
|---|---|---|
| `detect/` | detectors, `postprocess` | YOLO (ONNX / TensorRT / OpenVINO via device profile), per-class thresholds + class-aware NMS from YAML |
| `track/` | `Kalman`, `Tracker` | ByteTrack: high/low association, Kalman motion, tentative → confirmed (3 hits) → lost → removed |
| `reid/` | `embed()` | Appearance embedding, computed **only for detections the motion model failed to associate** in *gated* mode |
| `geometry/` | `Geometry` | Calibrated homography image <-> ground plane (edit the quads in the YAML) |
| `hazards/` | `HazardEngine` | Safety zone, time-to-contact, NONE / CAUTION / STOP, JSON stream + optional MQTT |
| `io/` | `World`, UI | Sources, overlay, publisher |
| `bench/` | `MOTEval`, `run_ablation` | MOTA / IDF1 / ID switches, ablation, TrackEval export |

### Honest scope notes
* On the real video the safety-zone homography is **not calibrated** to that camera (it uses the default quad); edit `calibration` in the YAML to match your camera. COCO-trained YOLOX-nano knows person / bicycle / car / dog only (no strollers or wheelchairs); the HOG fallback finds people only.
* The **simulated detector** emulates a YOLO model (jitter, misses, low-confidence drop-outs, false positives, occlusion behind bushes). Its latency is *not* a real inference latency. Select **YOLO (ultralytics)** with uploaded video or webcam to benchmark the real model on your device.
* COCO weights have no *stroller* or *wheelchair* classes. Fine-tune YOLOv8n / YOLO11n on your annotated sidewalk clips and point `detector.weights` at the result.
* `embed()` is a lightweight colour-layout embedding with the same interface as OSNet. Swap it for OSNet (e.g. torchreid) on real footage.
* In-app MOTA / IDF1 are quick approximations on **synthetic clips with perfect ground truth**. For official numbers (incl. **HOTA**) download the TrackEval export in the Benchmark tab.
* Pipeline FPS and latency cover detect → track → re-ID → hazards. Loop FPS also includes simulation, overlay drawing and streaming to your browser.

### Annotation guidelines (CVAT / Label Studio)
1. Classes: person, bicycle, car, dog, stroller, wheelchair (+ optional *other*). A cyclist = one *bicycle* box covering rider and bike.
2. One box per object, tight to the visible extent; keep the **same track ID** through occlusions up to 3 s.
3. Annotate objects ≥ 24 px tall; mark objects < 30% visible as *occluded* (ignored in evaluation).
4. Cover night, rain reflections, crowds, low sun, and partial occlusion. ≥ 10 clips of 10-30 s, 15-30 FPS.
5. Export **MOT 1.1**; double-annotate 10% of frames and check IoU agreement ≥ 0.8.

### Privacy
Faces are blurred (head region of every detected person) before display or recording. **No frames are stored by default**; the recorded demo clip is created only when you press the button and contains blurred output. Hazard events carry only IDs, class, distance and TTC - no imagery.

### Run the official evaluation
Unzip the export into your TrackEval `data/` folder, then:
`python scripts/run_mot_challenge.py --BENCHMARK SIDEWALK --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL SidewalkTrack-reid_gated --METRICS HOTA CLEAR Identity`
(All classes are exported as MOTChallenge class 1; check the options of your TrackEval version.)
"""


def main():
    st.set_page_config(page_title="SidewalkTrack", page_icon="🤖", layout="wide")
    st.markdown("<style>.block-container{padding-top:1.2rem}</style>", unsafe_allow_html=True)
    S = st.session_state
    S.setdefault("running", True)
    S.setdefault("yaml", DEFAULT_YAML)
    S.setdefault("pub", Publisher())
    S.setdefault("demo_path", None)
    for k in ("world", "sig", "last", "abl", "clip", "vpath", "vkey", "bench"):
        S.setdefault(k, None)

    # ---------------- sidebar ----------------
    with st.sidebar:
        st.title("🤖 SidewalkTrack")
        st.caption("Real-time detection, tracking & hazard logic for a sidewalk delivery robot")
        src = st.radio("Video source", ["Real pedestrian video (demo)", "Simulated sidewalk feed", "Upload video", "Webcam"])
        up = st.file_uploader("Video file", type=["mp4", "mov", "avi", "mkv"]) if src == "Upload video" else None
        scenario = st.selectbox("Scenario", list(SCEN), format_func=lambda s: {"day": "☀️ Day", "night": "🌙 Night", "rain": "🌧️ Rain reflections",
                                                                               "crowded": "👥 Crowded"}[s])
        res = st.selectbox("Resolution", ["640x360", "960x540", "1280x720"], index=1)
        fps = st.select_slider("Target FPS", [15, 20, 25, 30], value=20)
        cfg_tmp, _ = load_cfg(S.yaml)
        profile = st.selectbox("Device profile", list(cfg_tmp["device_profiles"]), format_func=lambda p: {
            "laptop_cpu": "Laptop CPU (ONNX)", "jetson": "NVIDIA Jetson (TensorRT)", "intel_nuc": "Intel NUC (OpenVINO)"}.get(p, p))
        det = st.selectbox("Detector", ["auto", "yolox", "hog", "yolo", "sim", "motion"], format_func=lambda d: {
            "auto": "Auto (best for the source)", "yolox": "YOLOX-nano ONNX (real, auto-download)",
            "hog": "OpenCV HOG people detector (real, built-in)", "yolo": "YOLO via ultralytics (v8n / 11n)",
            "sim": "Simulated YOLO-style detector", "motion": "Motion blobs (no model)"}[d])
        mode = st.selectbox("Tracker", list(MODES), index=2, format_func=lambda m: MODES[m])
        obey = st.toggle("Robot obeys hazards (brakes on STOP)", True)
        blur = st.toggle("Privacy: blur faces", True)
        show_lost = st.toggle("Show lost / coasting tracks", True)
        mq = st.toggle("Publish hazards to MQTT", False)
        with st.expander("⚙️ YAML config (thresholds, zone, calibration)"):
            S.yaml = st.text_area("config", S.yaml, height=330, label_visibility="collapsed")
            if st.button("Reset YAML"):
                S.yaml = DEFAULT_YAML
                st.rerun()
        st.download_button("⬇️ Hazard events (JSONL)", "\n".join(json.dumps(e) for e in S.pub.q) or "{}", "hazard_events.jsonl")
        st.caption(S.pub.status)

    cfg, err = load_cfg(S.yaml)
    if err:
        st.sidebar.error(f"YAML error, using defaults: {err}")
    if src == "Upload video" and up is not None:
        key = (up.name, up.size)
        if S.vkey != key:
            p = os.path.join(tempfile.mkdtemp(), up.name)
            with open(p, "wb") as f:
                f.write(up.getbuffer())
            S.vpath, S.vkey = p, key
    video_path = S.vpath
    if src.startswith("Real"):
        if S.demo_path is None:
            with st.spinner("Downloading demo pedestrian video (one time, ~8 MB)..."):
                S.demo_path = fetch_demo_video() or ""
        if S.demo_path:
            source, video_path = "video", S.demo_path
        else:
            source = "sim"
            st.sidebar.warning("Could not download the demo video (offline?). Showing the simulated feed - or upload your own video.")
    elif src == "Upload video" and S.vpath:
        source = "video"
    elif src == "Webcam":
        source = "webcam"
    else:
        source = "sim"
    W, H = (int(v) for v in res.split("x"))
    opts = dict(source=source, video_path=video_path, scenario=scenario, W=W, H=H, fps=fps, profile=profile, det=det, mode=mode,
                obey=obey, blur=blur, show_lost=show_lost)
    if mq and S.pub.client is None:
        S.pub.connect(cfg["mqtt"]["host"], cfg["mqtt"]["port"])
    sig = json.dumps([{k: v for k, v in opts.items() if k not in ("obey", "blur", "show_lost")}, S.yaml], sort_keys=True, default=str)
    if S.sig != sig or S.world is None:
        if S.world is not None:
            S.world.close()
        S.world, S.sig, S.last = World(cfg, opts, pub=S.pub), sig, None
    w = S.world
    w.o.update(obey=obey, blur=blur, show_lost=show_lost)

    tab_live, tab_bench, tab_about = st.tabs(["🎥 Live demo", "📊 Benchmark & ablation", "📖 Report & guide"])

    # ---------------- benchmark tab ----------------
    with tab_bench:
        st.subheader("1 · Single-frame detector benchmark")
        st.caption("Measures the detector selected in the sidebar. Use *YOLO* with a device profile to benchmark ONNX / TensorRT / OpenVINO.")
        if st.button("Run detector benchmark (40 frames)"):
            S.running = False
            with st.spinner("Benchmarking..."):
                ms, notes, name = bench_detector(cfg, opts)
            S.bench = dict(ms=ms, notes=notes, name=name, profile=profile)
        if S.bench:
            b = S.bench["ms"]
            st.dataframe(pd.DataFrame([{"Detector": S.bench["name"], "Profile": S.bench["profile"], "Mean ms": round(np.mean(b), 2),
                                        "p50 ms": round(np.percentile(b, 50), 2), "p95 ms": round(np.percentile(b, 95), 2),
                                        "FPS": round(1000 / max(np.mean(b), 1e-6), 1)}]), hide_index=True)
            for n in S.bench["notes"]:
                st.caption(n)
            if S.bench["name"] == "SimDetector":
                st.info("The simulated detector is not a real model - its timing is only the emulation cost.")
        st.subheader("2 · Ablation: ByteTrack vs + Re-ID vs + gated Re-ID")
        c1, c2 = st.columns(2)
        seeds = c1.slider("Clips per scenario (4 scenarios)", 1, 4, 3)
        frames = c2.slider("Frames per clip", 100, 400, 160, step=20)
        st.caption(f"Built-in test set = {4 * seeds} clips (day, night, rain reflections, crowded) with perfect ground truth.")
        if st.button("▶ Run ablation (pauses live view)", type="primary"):
            S.running = False
            bar = st.progress(0.0, text="Starting...")
            S.abl = run_ablation(cfg, seeds, frames, 20, bar)
        if S.abl:
            a = S.abl
            st.dataframe(a["summary"], hide_index=True)
            ch1, ch2 = st.columns(2)
            ch1.caption("Accuracy (IDF1)")
            ch1.bar_chart(a["summary"].set_index("Tracker")["IDF1"])
            ch2.caption("Tracker cost (ms/frame)")
            ch2.bar_chart(a["summary"].set_index("Tracker")["Tracker ms/frame"])
            st.caption("Speed-accuracy trade-off: gated embeddings keep most of the ID-switch reduction at a fraction of the embedding cost.")
            with st.expander("Per-scenario results"):
                st.dataframe(a["per"], hide_index=True)
            with st.expander("Annotated failure cases (ID switches)"):
                st.dataframe(a["fails"], hide_index=True) if len(a["fails"]) else st.write("No ID switches recorded.")
            st.download_button("⬇️ TrackEval export (GT + predictions, MOTChallenge layout)", a["zip"], "sidewalktrack_trackeval.zip")
        st.subheader("3 · Live profile (per-stage latency & memory)")
        m, pipe = w.profile()
        if m:
            st.dataframe(pd.DataFrame([{"Stage": k, "Mean ms": round(v, 2)} for k, v in m.items()]
                                      + [{"Stage": "PIPELINE (detect→hazards)", "Mean ms": round(pipe, 2)}]), hide_index=True)
            st.caption(f"Process memory: {rss_mb():.0f} MB")

    with tab_about:
        st.markdown(ABOUT)

    # ---------------- live tab ----------------
    with tab_live:
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("⏸ Pause" if S.running else "▶ Resume"):
            S.running = not S.running
            st.rerun()
        if b2.button("🔄 Reset scene"):
            S.sig = None
            st.rerun()
        if b3.button("🚶 Spawn pedestrian in path"):
            w.spawn_hazard()
        if b4.button("🎬 Record 10 s clip"):
            with st.spinner("Rendering annotated clip..."):
                S.clip = record_clip(cfg, opts)
        if S.clip:
            with st.expander("Last recorded clip (faces blurred)", expanded=True):
                st.video(S.clip)
                st.download_button("⬇️ Download MP4", S.clip, "sidewalktrack_demo.mp4", "video/mp4")
        for n in w.notes:
            st.caption("ℹ️ " + n)
        col_v, col_s = st.columns([2.2, 1])
        with col_v:
            ph_frame = st.empty()
        with col_s:
            ph_banner, ph_cards, ph_stage = st.empty(), st.empty(), st.empty()
        st.markdown("**Active tracks** (ID · class · velocity · age · time-to-lost) and **hazard events**")
        t1, t2 = st.columns([1.6, 1])
        with t1:
            ph_tracks = st.empty()
        with t2:
            ph_events = st.empty()

        tg = cfg["targets"]

        def render(vis, info, full=True):
            show_image(ph_frame, vis)
            if not full:
                return
            ov = info["overall"]
            col = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}[ov]
            ph_banner.markdown(f"<div style='padding:12px;border-radius:12px;background:{col};color:#111;font-size:1.5rem;font-weight:800;"
                               f"text-align:center'>{['CLEAR', 'CAUTION', 'STOP'][ov]} · robot {w.speed:.1f} m/s</div>", unsafe_allow_html=True)
            m, pipe = w.profile()
            loop = sum(m.values())
            pfps = 1000 / pipe if pipe > 0 else 0
            ph_cards.markdown(card_grid([
                ("Pipeline FPS", f"{pfps:.0f}", pfps >= tg["fps"]), ("Capture→hazard", f"{pipe:.1f} ms", pipe <= tg["latency_ms"]),
                ("Tracks", info["n_tracks"], None), ("Embeds/frame", f"{np.mean(w.nemb):.2f}" if w.nemb else "0", None),
                ("Loop FPS", f"{1000 / loop:.0f}" if loop else "-", None), ("Memory", f"{rss_mb():.0f} MB", None)]), unsafe_allow_html=True)
            md = "| Stage | ms |\n|---|---|\n" + "\n".join(f"| {k} | {v:.2f} |" for k, v in m.items())
            ph_stage.markdown(md)
            ph_tracks.dataframe(pd.DataFrame(info["rows"]) if info["rows"] else pd.DataFrame({"ID": []}), hide_index=True, height=230)
            ph_events.dataframe(pd.DataFrame(list(w.events)[:8])[["iso", "track_id", "cls", "risk", "ttc_s", "dist_m"]]
                                if w.events else pd.DataFrame({"iso": [], "track_id": [], "cls": [], "risk": []}), hide_index=True, height=230)

        if not S.running:
            if S.last is None:
                S.last = w.step()
            render(*S.last)
        else:
            n = 0
            while S.running:
                t_start = time.perf_counter()
                vis, info = w.step()
                S.last = (vis, info)
                render(vis, info, full=(w.frame % 4 == 0 or w.frame < 3))
                n += 1
                if MAX_FRAMES and n >= MAX_FRAMES:
                    break
                time.sleep(max(0.0, 1.0 / w.fps - (time.perf_counter() - t_start)))


if __name__ == "__main__":
    main()