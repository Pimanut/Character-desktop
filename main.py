import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import threading
import subprocess
import importlib
import sys
import os
import io
import re
import json
import struct
import base64
import urllib.parse
import random
import math
import time
import gc


def ensure(module, package):
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", package],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            importlib.invalidate_caches()
            importlib.import_module(module)
            return True
        except Exception:
            return False


if not all(ensure(m, p) for m, p in (("PIL", "Pillow"), ("numpy", "numpy"), ("psutil", "psutil"))):
    _root = tk.Tk()
    _root.withdraw()
    messagebox.showerror("Error", "ติดตั้งไลบรารีไม่สำเร็จ\nรัน: python -m pip install Pillow numpy psutil")
    sys.exit(1)

import numpy as np
import psutil
from PIL import Image, ImageDraw, ImageFilter, ImageTk, ImageChops

CHARACTER_SCALE = 0.3
CHARACTER_MAX = 512
TRANS_KEY = "#000001"
KEY_RGBA = (0, 0, 1, 255)
SPIN_STEP = 6.0
WIN_W, WIN_H = 560, 612

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(BASE_DIR, "assets")
IMAGE_EXT = {".png", ".apng", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
MODEL_EXT = {".glb", ".gltf", ".obj"}
FILE_TYPES = [
    ("Supported files", "*.png *.apng *.jpg *.jpeg *.gif *.webp *.bmp "
                        "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.wmv *.m4v "
                        "*.glb *.gltf *.obj"),
    ("All files", "*.*"),
]
YOUTUBE_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([\w\-]{11})")
LIGHT = np.array([0.35, 0.55, 0.75], np.float32)
LIGHT /= np.linalg.norm(LIGHT)

for _sub in ("images", "videos", "models"):
    try:
        os.makedirs(os.path.join(ASSETS, _sub), exist_ok=True)
    except OSError:
        pass


def is_url(text):
    return text.startswith(("http://", "https://", "www."))


def is_youtube_url(text):
    return bool(YOUTUBE_PATTERN.search(text))


def fit_size(w, h):
    if w > CHARACTER_MAX or h > CHARACTER_MAX:
        ratio = min(CHARACTER_MAX / w, CHARACTER_MAX / h)
        w, h = w * ratio, h * ratio
    return max(1, int(w * CHARACTER_SCALE)), max(1, int(h * CHARACTER_SCALE))


def start_dir():
    return ASSETS if os.path.isdir(ASSETS) else BASE_DIR


class Governor:
    def __init__(self):
        self.proc = psutil.Process()
        self.cores = psutil.cpu_count() or 1
        self.cpu = 50
        self.ram = 1024
        self.gpu = 60
        self.scale = 1.0
        self.cpu_now = 0.0
        self.rss = 0.0
        self.proc.cpu_percent(None)

    def poll(self):
        self.cpu_now = self.proc.cpu_percent(None) / self.cores
        self.rss = self.proc.memory_info().rss / 1048576
        if self.cpu_now > self.cpu:
            self.scale = min(8.0, self.scale * 1.3)
        elif self.cpu_now < self.cpu * 0.7:
            self.scale = max(1.0, self.scale * 0.9)

    def over_ram(self):
        return self.rss > self.ram

    def fps_cap(self):
        return max(5, int(60 * self.gpu / 100))

    def delay(self, ms):
        return max(ms, 1000 / self.fps_cap()) * self.scale

    def face_budget(self):
        return int(3000 + 17000 * self.gpu / 100)


GOV = Governor()


COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
             5123: np.uint16, 5125: np.uint32, 5126: np.float32}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}


def read_gltf(path):
    with open(path, "rb") as f:
        data = f.read()
    blob = None
    if data[:4] == b"glTF":
        doc = None
        offset = 12
        while offset + 8 <= len(data):
            size, kind = struct.unpack_from("<II", data, offset)
            offset += 8
            chunk = data[offset:offset + size]
            offset += size
            if kind == 0x4E4F534A:
                doc = json.loads(chunk.decode("utf-8"))
            elif kind == 0x004E4942:
                blob = chunk
    else:
        doc = json.loads(data.decode("utf-8-sig"))
    folder = os.path.dirname(path)
    buffers = []
    for b in doc.get("buffers", []):
        uri = b.get("uri")
        if uri is None:
            buffers.append(blob)
        elif uri.startswith("data:"):
            buffers.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            with open(os.path.join(folder, urllib.parse.unquote(uri)), "rb") as f:
                buffers.append(f.read())
    return doc, buffers, folder


def read_accessor(doc, buffers, index):
    acc = doc["accessors"][index]
    dtype = np.dtype(COMPONENT[acc["componentType"]])
    width = WIDTH[acc["type"]]
    count = acc["count"]
    if "bufferView" not in acc:
        return np.zeros((count, width), np.float32)
    view = doc["bufferViews"][acc["bufferView"]]
    buf = buffers[view["buffer"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", 0)
    item = dtype.itemsize * width
    if stride and stride != item:
        raw = np.frombuffer(buf, np.uint8, stride * (count - 1) + item, start)
        arr = raw[np.arange(count)[:, None] * stride + np.arange(item)[None, :]].copy().view(dtype)
    else:
        arr = np.frombuffer(buf, dtype, count * width, start)
    arr = arr.reshape(count, width)
    if acc.get("normalized"):
        arr = np.maximum(arr.astype(np.float32) / np.iinfo(dtype).max, -1.0)
    return arr


def load_texture(doc, buffers, folder, index):
    src = doc["images"][doc["textures"][index]["source"]]
    if "bufferView" in src:
        view = doc["bufferViews"][src["bufferView"]]
        off = view.get("byteOffset", 0)
        raw = buffers[view["buffer"]][off:off + view["byteLength"]]
    elif src["uri"].startswith("data:"):
        raw = base64.b64decode(src["uri"].split(",", 1)[1])
    else:
        with open(os.path.join(folder, urllib.parse.unquote(src["uri"])), "rb") as f:
            raw = f.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    img.thumbnail((256, 256))
    return np.asarray(img, np.float32) / 255.0


def new_node(mesh=None):
    return {"t": np.zeros(3, np.float32), "r": np.array([0, 0, 0, 1], np.float32),
            "s": np.ones(3, np.float32), "m": None, "children": [], "mesh": mesh, "skin": None}


def quat_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0],
        [0, 0, 0, 1]], np.float32)


def compose(t, r, s):
    m = quat_matrix(r)
    m[:3, :3] *= s[None, :]
    m[:3, 3] = t
    return m


def sample(times, values, t, mode, rotation):
    if len(times) == 1 or t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    i = int(np.searchsorted(times, t, side="right")) - 1
    if mode == "STEP":
        return values[i]
    f = (t - times[i]) / (times[i + 1] - times[i])
    a, b = values[i], values[i + 1]
    if rotation:
        if float(np.dot(a, b)) < 0:
            b = -b
        r = a * (1 - f) + b * f
        return r / np.linalg.norm(r)
    return a * (1 - f) + b * f


class Model:
    def __init__(self, meshes, nodes, roots, skins, anims):
        self.meshes, self.nodes, self.roots = meshes, nodes, roots
        self.skins, self.anims = skins, anims
        faces = sum(len(p["tri"]) for m in meshes for p in m)
        if faces == 0:
            raise ValueError("ไม่พบพื้นผิวของโมเดล")
        if faces > 400000:
            raise ValueError("โมเดลใหญ่เกินไป (เกิน 400,000 faces)")
        points = self.vertices(self.pose(None, 0.0))[0]
        lo, hi = points.min(axis=0), points.max(axis=0)
        self.center = (lo + hi) / 2
        self.scale = 2.0 / max(float((hi - lo).max()), 1e-6)

    def pose(self, anim, t):
        override = {}
        if anim is not None:
            for node, path, times, values, mode in self.anims[anim]["channels"]:
                override.setdefault(node, {})[path] = sample(times, values, t, mode, path == "rotation")
        world = [None] * len(self.nodes)
        stack = [(r, np.eye(4, dtype=np.float32)) for r in self.roots]
        while stack:
            i, parent = stack.pop()
            node = self.nodes[i]
            o = override.get(i, {})
            if node["m"] is not None and not o:
                local = node["m"]
            else:
                local = compose(o.get("translation", node["t"]), o.get("rotation", node["r"]),
                                o.get("scale", node["s"]))
            world[i] = parent @ local
            for c in node["children"]:
                stack.append((c, world[i]))
        return world

    def vertices(self, world):
        pts, tris, cols, dbl, offset = [], [], [], [], 0
        eye = np.eye(4, dtype=np.float32)
        for ni, node in enumerate(self.nodes):
            if node["mesh"] is None or world[ni] is None:
                continue
            for part in self.meshes[node["mesh"]]:
                pos = part["pos"]
                homo = np.concatenate([pos, np.ones((len(pos), 1), np.float32)], axis=1)
                if node["skin"] is not None and part["joints"] is not None:
                    skin = self.skins[node["skin"]]
                    mats = np.stack([(world[j] if world[j] is not None else eye) @ skin["ibm"][k]
                                     for k, j in enumerate(skin["joints"])])
                    out = np.zeros((len(pos), 4), np.float32)
                    for k in range(4):
                        out += part["weights"][:, k:k + 1] * np.einsum(
                            "nij,nj->ni", mats[part["joints"][:, k]], homo)
                    moved = out[:, :3]
                else:
                    moved = (homo @ world[ni].T)[:, :3]
                pts.append(moved)
                tris.append(part["tri"] + offset)
                cols.append(part["col"])
                dbl.append(np.full(len(part["tri"]), part["dbl"]))
                offset += len(pos)
        if not pts:
            raise ValueError("ไม่พบพื้นผิวของโมเดล")
        return (np.concatenate(pts), np.concatenate(tris), np.concatenate(cols), np.concatenate(dbl))

    def render(self, w, h, yaw, pitch, zoom, anim, t, budget):
        points, tri, colors, double = self.vertices(self.pose(anim, t))
        points = (points - self.center) * self.scale
        cy, sy, cx, sx = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], np.float32)
        rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], np.float32)
        view = points @ (rx @ ry).T
        ss = 2 if w * h <= 160000 else 1
        W, H = w * ss, h * ss
        k = min(W, H) * 0.45 * zoom
        screen = np.stack([W / 2 + view[:, 0] * k, H / 2 - view[:, 1] * k], axis=1)
        v = view[tri]
        n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
        length = np.linalg.norm(n, axis=1)
        length[length == 0] = 1
        n = n / length[:, None]
        s = screen[tri]
        area = 0.5 * np.abs((s[:, 1, 0] - s[:, 0, 0]) * (s[:, 2, 1] - s[:, 0, 1])
                            - (s[:, 2, 0] - s[:, 0, 0]) * (s[:, 1, 1] - s[:, 0, 1]))
        idx = np.nonzero(((n[:, 2] > 0) | double) & (area > 0.05))[0]
        if len(idx) > budget:
            idx = idx[np.argpartition(-area[idx], budget)[:budget]]
        idx = idx[np.argsort(v[idx, :, 2].mean(axis=1))]
        shade = 0.3 + 0.7 * np.abs(n[idx] @ LIGHT)
        rgb = (np.clip(colors[idx] * shade[:, None], 0, 1) * 255).astype(np.uint8)
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for poly, c in zip(s[idx].reshape(-1, 6).tolist(), rgb.tolist()):
            draw.polygon(poly, fill=(c[0], c[1], c[2], 255))
        if ss > 1:
            img = img.resize((w, h), Image.LANCZOS)
        return img


def load_gltf(path):
    doc, buffers, folder = read_gltf(path)
    textures = {}
    meshes = []
    for mesh in doc.get("meshes", []):
        parts = []
        for prim in mesh["primitives"]:
            attrs = prim["attributes"]
            if prim.get("mode", 4) != 4 or "POSITION" not in attrs:
                continue
            pos = read_accessor(doc, buffers, attrs["POSITION"]).astype(np.float32)
            if "indices" in prim:
                idx = read_accessor(doc, buffers, prim["indices"]).reshape(-1).astype(np.int64)
            else:
                idx = np.arange(len(pos))
            tri = idx[:len(idx) // 3 * 3].reshape(-1, 3)
            material = doc["materials"][prim["material"]] if "material" in prim else {}
            pbr = material.get("pbrMetallicRoughness", {})
            base = np.array(pbr.get("baseColorFactor", [1, 1, 1, 1]), np.float32)[:3] ** (1 / 2.2)
            colors = np.tile(base, (len(pos), 1))
            if "COLOR_0" in attrs:
                colors = colors * read_accessor(doc, buffers, attrs["COLOR_0"])[:, :3].astype(np.float32)
            info = pbr.get("baseColorTexture")
            if info is not None and "TEXCOORD_0" in attrs:
                key = info["index"]
                if key not in textures:
                    try:
                        textures[key] = load_texture(doc, buffers, folder, key)
                    except Exception:
                        textures[key] = None
                tex = textures[key]
                if tex is not None:
                    uv = read_accessor(doc, buffers, attrs["TEXCOORD_0"]).astype(np.float32)
                    th, tw = tex.shape[:2]
                    xs = (np.mod(uv[:, 0], 1.0) * (tw - 1)).astype(np.int64)
                    ys = (np.mod(uv[:, 1], 1.0) * (th - 1)).astype(np.int64)
                    colors = colors * tex[ys, xs]
            part = {"pos": pos, "tri": tri, "col": colors[tri].mean(axis=1),
                    "dbl": bool(material.get("doubleSided", False)), "joints": None, "weights": None}
            if "JOINTS_0" in attrs and "WEIGHTS_0" in attrs:
                part["joints"] = read_accessor(doc, buffers, attrs["JOINTS_0"]).astype(np.int64)
                wts = read_accessor(doc, buffers, attrs["WEIGHTS_0"]).astype(np.float32)
                total = wts.sum(axis=1, keepdims=True)
                total[total == 0] = 1
                part["weights"] = wts / total
            parts.append(part)
        meshes.append(parts)

    nodes = []
    for n in doc.get("nodes", []):
        node = new_node(n.get("mesh"))
        node["skin"] = n.get("skin")
        node["children"] = n.get("children", [])
        if "translation" in n:
            node["t"] = np.array(n["translation"], np.float32)
        if "rotation" in n:
            node["r"] = np.array(n["rotation"], np.float32)
        if "scale" in n:
            node["s"] = np.array(n["scale"], np.float32)
        if "matrix" in n:
            node["m"] = np.array(n["matrix"], np.float32).reshape(4, 4).T
        nodes.append(node)
    if not nodes:
        nodes = [new_node(i) for i in range(len(meshes))]
        roots = list(range(len(nodes)))
    else:
        roots = None
        if doc.get("scenes"):
            roots = doc["scenes"][doc.get("scene", 0)].get("nodes")
        if roots is None:
            child = {c for n in nodes for c in n["children"]}
            roots = [i for i in range(len(nodes)) if i not in child]

    skins = []
    for s in doc.get("skins", []):
        joints = s["joints"]
        if "inverseBindMatrices" in s:
            ibm = read_accessor(doc, buffers, s["inverseBindMatrices"]).astype(np.float32)
            ibm = ibm.reshape(-1, 4, 4).transpose(0, 2, 1)
        else:
            ibm = np.tile(np.eye(4, dtype=np.float32), (len(joints), 1, 1))
        skins.append({"joints": joints, "ibm": ibm})

    anims = []
    for a in doc.get("animations", []):
        channels = []
        for c in a["channels"]:
            target = c["target"]
            if "node" not in target or target["path"] not in ("translation", "rotation", "scale"):
                continue
            sampler = a["samplers"][c["sampler"]]
            times = read_accessor(doc, buffers, sampler["input"]).reshape(-1).astype(np.float32)
            values = read_accessor(doc, buffers, sampler["output"]).astype(np.float32)
            mode = sampler.get("interpolation", "LINEAR")
            if mode == "CUBICSPLINE":
                values = values[1::3]
            channels.append((target["node"], target["path"], times, values, mode))
        if channels:
            anims.append({"name": a.get("name", ""), "channels": channels,
                          "duration": max(float(ch[2][-1]) for ch in channels)})
    return Model(meshes, nodes, roots, skins, anims)


def read_mtl(path):
    colors, name = {}, None
    try:
        with open(path, "r", errors="ignore") as f:
            for line in f:
                p = line.split()
                if not p:
                    continue
                if p[0] == "newmtl":
                    name = " ".join(p[1:])
                elif p[0] == "Kd" and name and len(p) >= 4:
                    colors[name] = tuple(float(x) for x in p[1:4])
    except OSError:
        pass
    return colors


def load_obj(path):
    verts, faces, face_cols = [], [], []
    mats, current = {}, (0.8, 0.8, 0.8)
    folder = os.path.dirname(path)
    with open(path, "r", errors="ignore") as f:
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "v" and len(p) >= 4:
                verts.append([float(x) for x in p[1:4]])
            elif p[0] == "mtllib":
                mats.update(read_mtl(os.path.join(folder, " ".join(p[1:]))))
            elif p[0] == "usemtl":
                current = mats.get(" ".join(p[1:]), (0.8, 0.8, 0.8))
            elif p[0] == "f":
                ids = []
                for tok in p[1:]:
                    i = int(tok.split("/")[0])
                    ids.append(i - 1 if i > 0 else len(verts) + i)
                for j in range(1, len(ids) - 1):
                    faces.append((ids[0], ids[j], ids[j + 1]))
                    face_cols.append(current)
    if not verts or not faces:
        raise ValueError("ไฟล์ OBJ ไม่มีข้อมูลพื้นผิว")
    part = {"pos": np.array(verts, np.float32), "tri": np.array(faces, np.int64),
            "col": np.array(face_cols, np.float32), "dbl": True, "joints": None, "weights": None}
    return Model([[part]], [new_node(0)], [0], [], [])


def load_model(path):
    if os.path.splitext(path)[1].lower() == ".obj":
        return load_obj(path)
    return load_gltf(path)


class ImageSource:
    interval = 20

    def __init__(self, path):
        self.error = None
        self.img = Image.open(path)
        self.frames = getattr(self.img, "n_frames", 1)
        self.size = fit_size(*self.img.size)
        self.interval = 20 if self.frames > 1 else 200
        self.index = 0
        self.next_at = 0.0
        self.stale = True
        self.shown = None
        self.paused = False

    def frame(self, w, h):
        now = time.time()
        if self.frames > 1 and not self.paused and now >= self.next_at and self.next_at:
            self.index = (self.index + 1) % self.frames
            self.stale = True
        if not self.stale and self.shown == (w, h):
            return None
        self.img.seek(self.index)
        self.next_at = now + max(20, self.img.info.get("duration", 100)) / 1000
        self.stale = False
        self.shown = (w, h)
        method = Image.BILINEAR if self.frames > 1 else Image.LANCZOS
        return self.img.convert("RGBA").resize((w, h), method)

    def toggle_pause(self):
        self.paused = not self.paused

    def menu_items(self):
        return [("Pause / Resume", self.toggle_pause)] if self.frames > 1 else []

    def close(self):
        try:
            self.img.close()
        except Exception:
            pass


class VideoSource:
    interval = 15

    def __init__(self, src):
        self.src = src
        self.error = None
        self.size = None
        self.latest = None
        self.target = (1, 1)
        self.lock = threading.Lock()
        self.running = True
        self.paused = False
        self.rewind = False
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        import cv2
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened():
            self.error = "เปิดวิดีโอไม่ได้:\n" + self.src
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps < 1 or fps > 120:
            fps = 30
        ok, first = cap.read()
        if not ok:
            self.error = "อ่านเฟรมแรกไม่ได้"
            return
        self.size = fit_size(first.shape[1], first.shape[0])
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
            if self.rewind:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.rewind = False
            start = time.time()
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue
            tw, th = self.target
            rgb = cv2.cvtColor(cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
            with self.lock:
                self.latest = rgb
            wait = GOV.delay(1000 / fps) / 1000 - (time.time() - start)
            if wait > 0:
                time.sleep(wait)
        cap.release()

    def frame(self, w, h):
        self.target = (max(1, w), max(1, h))
        with self.lock:
            rgb, self.latest = self.latest, None
        if rgb is None:
            return None
        return Image.fromarray(rgb).convert("RGBA")

    def toggle_pause(self):
        self.paused = not self.paused

    def restart(self):
        self.rewind = True
        self.paused = False

    def menu_items(self):
        return [("Pause / Resume", self.toggle_pause), ("Restart", self.restart)]

    def close(self):
        self.running = False


class ModelSource:
    interval = 33

    def __init__(self, path):
        self.error = None
        self.size = None
        self.model = None
        self.yaw, self.pitch, self.zoom = 0.0, 0.25, 1.0
        self.turntable = True
        self.anim = None
        self.playing = True
        self.clock = 0.0
        self.last = time.time()
        self.dirty = True
        self.last_size = None
        threading.Thread(target=self._load, args=(path,), daemon=True).start()

    def _load(self, path):
        try:
            model = load_model(path)
            if model.anims:
                self.anim = 0
                self.turntable = False
            self.model = model
            self.size = fit_size(CHARACTER_MAX, CHARACTER_MAX)
        except Exception as e:
            self.error = "โหลดโมเดลไม่สำเร็จ:\n" + str(e)

    def frame(self, w, h):
        now = time.time()
        dt = now - self.last
        self.last = now
        if (w, h) != self.last_size:
            self.last_size = (w, h)
            self.dirty = True
        moving = False
        if self.turntable:
            self.yaw += dt * 0.9
            moving = True
        if self.anim is not None and self.playing:
            self.clock += dt
            moving = True
        if not (moving or self.dirty):
            return None
        self.dirty = False
        duration = self.model.anims[self.anim]["duration"] if self.anim is not None else 0
        t = self.clock % duration if duration > 0 else 0.0
        return self.model.render(w, h, self.yaw, self.pitch, self.zoom, self.anim, t, GOV.face_budget())

    def rotate(self, dx, dy):
        self.yaw += dx * 0.02
        self.pitch = max(-1.4, min(1.4, self.pitch + dy * 0.02))
        self.dirty = True

    def wheel(self, delta):
        self.zoom = max(0.3, min(3.0, self.zoom * (1.1 if delta > 0 else 1 / 1.1)))
        self.dirty = True

    def toggle_turntable(self):
        self.turntable = not self.turntable
        self.dirty = True

    def toggle_play(self):
        self.playing = not self.playing

    def next_anim(self):
        if self.model and self.model.anims:
            if self.anim is None:
                self.anim = 0
            else:
                self.anim = (self.anim + 1) % len(self.model.anims)
            self.clock = 0.0
            self.playing = True

    def reset_view(self):
        self.yaw, self.pitch, self.zoom = 0.0, 0.25, 1.0
        self.dirty = True

    def menu_items(self):
        items = [("Toggle Turntable", self.toggle_turntable)]
        if self.model and self.model.anims:
            items.append(("Play / Pause Animation", self.toggle_play))
            if len(self.model.anims) > 1:
                items.append(("Next Animation", self.next_anim))
        items.append(("Reset View", self.reset_view))
        return items

    def close(self):
        self.model = None


def make_source(path):
    ext = os.path.splitext(path.split("?")[0])[1].lower()
    if ext in MODEL_EXT:
        return ModelSource(path)
    if is_url(path) or ext in VIDEO_EXT:
        if not ensure("cv2", "opencv-python"):
            raise RuntimeError("ติดตั้ง opencv-python ไม่สำเร็จ\nรัน: pip install opencv-python")
        return VideoSource(path)
    return ImageSource(path)


class AIWalker:
    TICK = 16
    SPEED_MIN = 1.5
    SPEED_MAX = 3.5
    STATES = ("walk", "walk", "walk", "idle", "wander", "circle")

    def __init__(self, cw):
        self.cw = cw
        self.running = True
        self.enabled = True
        self.hold = False
        self.state = "walk"
        self.angle = random.uniform(0, 2 * math.pi)
        self.speed = 0.0
        self.left = 0
        self.dir = 1
        self.after_id = None
        self._next()
        cw.master.after(100, self._loop)

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def stop(self):
        self.running = False
        if self.after_id:
            try:
                self.cw.master.after_cancel(self.after_id)
            except tk.TclError:
                pass

    def _next(self):
        self.state = random.choice(self.STATES)
        if self.state == "walk":
            self.angle += random.uniform(-math.pi / 2, math.pi / 2)
            self.speed = random.uniform(self.SPEED_MIN, self.SPEED_MAX)
            self.left = random.randint(60, 220)
        elif self.state == "idle":
            self.speed = 0.0
            self.left = random.randint(30, 120)
        elif self.state == "wander":
            self.speed = random.uniform(self.SPEED_MIN, self.SPEED_MAX * 0.7)
            self.left = random.randint(80, 180)
        else:
            self.speed = random.uniform(self.SPEED_MIN, self.SPEED_MAX * 0.8)
            self.dir = random.choice([-1, 1])
            self.left = random.randint(120, 300)

    def _loop(self):
        if not self.running:
            return
        delay = GOV.delay(self.TICK)
        try:
            if self.enabled and not self.hold:
                self._step(delay / self.TICK)
            self.after_id = self.cw.master.after(int(delay), self._loop)
        except tk.TclError:
            self.running = False

    def _step(self, k):
        win = self.cw.win
        sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
        cw, ch = win.winfo_width(), win.winfo_height()
        if self.state == "wander":
            self.angle += random.uniform(-0.05, 0.05)
        elif self.state == "circle":
            self.angle += self.dir * 0.09 * k
        if self.state != "idle":
            nx = win.winfo_x() + math.cos(self.angle) * self.speed * k
            ny = win.winfo_y() + math.sin(self.angle) * self.speed * k
            bounced = False
            if nx < 0:
                nx, self.angle, bounced = 0, math.pi - self.angle, True
            elif nx + cw > sw:
                nx, self.angle, bounced = sw - cw, math.pi - self.angle, True
            if ny < 0:
                ny, self.angle, bounced = 0, -self.angle, True
            elif ny + ch > sh:
                ny, self.angle, bounced = sh - ch, -self.angle, True
            if bounced:
                self.angle += random.uniform(-0.3, 0.3)
            win.geometry(f"+{int(nx)}+{int(ny)}")
        self.left -= 1
        if self.left <= 0:
            self._next()


class CharacterWindow:
    def __init__(self, master, app, source, path, on_close=None):
        self.master, self.app, self.source, self.path = master, app, source, path
        self.on_close = on_close
        self.alive = True
        self.placed = False
        self.walker = None
        self.spinning = False
        self.spin_angle = 0.0
        self.w = self.h = 0
        self.current = None
        self.photo = None
        self.after_id = None
        self.dx = self.dy = 0

        self.win = tk.Toplevel(master)
        self.win.overrideredirect(True)
        self.win.wm_attributes("-topmost", True)
        self.win.withdraw()
        self.lbl = tk.Label(self.win, bd=0, highlightthickness=0, bg=TRANS_KEY)
        self.lbl.pack(fill="both", expand=True)
        for widget in (self.win, self.lbl):
            widget.bind("<ButtonPress-1>", self._press)
            widget.bind("<B1-Motion>", self._motion)
            widget.bind("<ButtonRelease-1>", self._release)
            widget.bind("<Button-3>", self._show_menu)
            widget.bind("<MouseWheel>", self._wheel)
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.menu = tk.Menu(self.win, tearoff=0, font=("Segoe UI", 10), bg="#0b2a55", fg="#eaf6ff",
                            activebackground="#5cb8ff", activeforeground="#06183a")
        self._build_menu()
        self.refresh_cursor()
        self._boot()

    def _build_menu(self):
        self.menu.delete(0, "end")
        self.menu.add_command(label="Change File", command=self._change)
        self.menu.add_command(label="Reload", command=lambda: self._swap(self.path))
        extras = self.source.menu_items()
        if extras:
            self.menu.add_separator()
            for label, command in extras:
                self.menu.add_command(label=label, command=command)
        self.menu.add_separator()
        self.menu.add_command(label="Resize", command=self._resize)
        self.menu.add_separator()
        self.menu.add_command(label="Start Spinning", command=self._toggle_spin)
        self.spin_index = self.menu.index("end")
        self.menu.add_command(label="Start Walking", command=self._toggle_walk)
        self.walk_index = self.menu.index("end")
        self.menu.add_separator()
        self.menu.add_command(label="Close", command=self.close)

    def refresh_cursor(self):
        self.lbl.configure(cursor="X_cursor" if self.app.delete_mode else "fleur")

    def _boot(self):
        if not self.alive:
            return
        src = self.source
        if src.error:
            messagebox.showerror("Error", src.error)
            self.close()
            return
        if src.size is None:
            self.after_id = self.win.after(50, self._boot)
            return
        self.w, self.h = src.size
        if self.placed:
            self.win.geometry(f"{self.w}x{self.h}")
        else:
            sw, sh = self.win.winfo_screenwidth(), self.win.winfo_screenheight()
            x = (sw - self.w) // 2 + random.randint(-80, 80)
            y = (sh - self.h) // 2 + random.randint(-60, 60)
            self.win.geometry(f"{self.w}x{self.h}+{x}+{y}")
            self.placed = True
        self.win.deiconify()
        try:
            self.win.wm_attributes("-transparentcolor", TRANS_KEY)
        except tk.TclError:
            pass
        self._loop()

    def _loop(self):
        if not self.alive:
            return
        changed = False
        try:
            img = self.source.frame(self.w, self.h)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.close()
            return
        if img is not None:
            self.current = img
            changed = True
        if self.spinning:
            self.spin_angle = (self.spin_angle + SPIN_STEP) % 360
            changed = True
        if changed and self.current is not None:
            self._paint()
        base = 16 if self.spinning else self.source.interval
        self.after_id = self.win.after(int(GOV.delay(base)), self._loop)

    def _paint(self):
        img = self.current
        if self.spinning:
            img = img.rotate(-self.spin_angle, resample=Image.BICUBIC)
        canvas = Image.alpha_composite(Image.new("RGBA", img.size, KEY_RGBA), img)
        self.photo = ImageTk.PhotoImage(canvas.convert("RGB"))
        self.lbl.configure(image=self.photo)

    def _swap(self, path):
        try:
            new = make_source(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        if self.after_id:
            try:
                self.win.after_cancel(self.after_id)
            except tk.TclError:
                pass
        self.source.close()
        self.source, self.path, self.current = new, path, None
        self._build_menu()
        self._boot()

    def _change(self):
        path = filedialog.askopenfilename(title="Select File", filetypes=FILE_TYPES,
                                          initialdir=start_dir())
        if path:
            self._swap(path)

    def _resize(self):
        val = simpledialog.askstring("Resize", "ใส่ขนาด เช่น 200x200:",
                                     initialvalue=f"{self.w}x{self.h}", parent=self.win)
        if val and "x" in val.lower():
            try:
                w, h = map(int, val.lower().split("x"))
            except ValueError:
                messagebox.showerror("Error", "รูปแบบไม่ถูกต้อง เช่น 200x200")
                return
            self.w, self.h = max(16, min(w, 2000)), max(16, min(h, 2000))
            self.win.geometry(f"{self.w}x{self.h}")

    def _toggle_spin(self):
        self.spinning = not self.spinning
        if not self.spinning and self.current is not None:
            self._paint()

    def _toggle_walk(self):
        if self.walker is None:
            self.walker = AIWalker(self)
        else:
            self.walker.toggle()

    def _show_menu(self, e):
        self.menu.entryconfig(self.spin_index, label="Stop Spinning" if self.spinning else "Start Spinning")
        walking = self.walker is not None and self.walker.enabled
        self.menu.entryconfig(self.walk_index, label="Stop Walking" if walking else "Start Walking")
        try:
            self.menu.tk_popup(e.x_root, e.y_root)
        finally:
            self.menu.grab_release()

    def _press(self, e):
        if self.app.delete_mode:
            self.close()
            return
        self.dx, self.dy = e.x, e.y
        if self.walker:
            self.walker.hold = True

    def _motion(self, e):
        if self.app.delete_mode:
            return
        if e.state & 1 and hasattr(self.source, "rotate"):
            self.source.rotate(e.x - self.dx, e.y - self.dy)
            self.dx, self.dy = e.x, e.y
            return
        x = self.win.winfo_x() + (e.x - self.dx)
        y = self.win.winfo_y() + (e.y - self.dy)
        self.win.geometry(f"+{x}+{y}")

    def _release(self, e):
        if self.walker:
            self.walker.hold = False
            self.walker.angle = random.uniform(0, 2 * math.pi)

    def _wheel(self, e):
        if hasattr(self.source, "wheel"):
            self.source.wheel(e.delta)

    def close(self):
        if not self.alive:
            return
        self.alive = False
        if self.after_id:
            try:
                self.win.after_cancel(self.after_id)
            except tk.TclError:
                pass
        if self.walker:
            self.walker.stop()
        self.source.close()
        if self.on_close:
            self.on_close(self)
        try:
            self.win.destroy()
        except tk.TclError:
            pass


def rounded_mask(w, h, r, ss=4):
    m = Image.new("L", (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), max(0, r) * ss, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def glass(base, box, radius, tint, top, bottom, blur=10):
    x, y, w, h = box
    layer = base.crop((x, y, x + w, y + h)).filter(ImageFilter.GaussianBlur(blur)).convert("RGBA")
    ramp = np.linspace(top, bottom, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    tinted = Image.new("RGBA", (w, h), tint + (0,))
    tinted.putalpha(Image.fromarray(ramp.astype(np.uint8)))
    layer = Image.alpha_composite(layer, tinted)

    gloss = np.zeros((h, w), np.float32)
    band = max(2, int(h * 0.5))
    gloss[:band] = np.linspace(75, 0, band, dtype=np.float32)[:, None]
    inset = Image.new("L", (w, h), 0)
    inset.paste(rounded_mask(max(1, w - 10), max(1, h - 6), radius - 3), (5, 3))
    shine = ImageChops.multiply(Image.fromarray(gloss.astype(np.uint8)), inset)
    white = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    white.putalpha(shine)
    layer = Image.alpha_composite(layer, white)

    mask = rounded_mask(w, h, radius)
    inner = Image.new("L", (w, h), 0)
    inner.paste(rounded_mask(max(1, w - 4), max(1, h - 4), radius - 2), (2, 2))
    ring = ImageChops.subtract(mask, inner).point(lambda v: int(v * 0.55))
    edge = Image.new("RGBA", (w, h), (255, 255, 255, 0))
    edge.putalpha(ring)
    layer = Image.alpha_composite(layer, edge)

    out = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    out.paste(layer, (0, 0), mask)
    return out


def make_background(w, h):
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    t = np.clip(yy * 0.8 + xx * 0.25, 0, 1)[..., None]
    top = np.array([5, 16, 44], np.float32)
    bottom = np.array([34, 108, 196], np.float32)
    img = Image.fromarray((top + (bottom - top) * t).astype(np.uint8)).convert("RGBA")

    glow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    g = ImageDraw.Draw(glow)
    g.ellipse((-80, -60, 260, 240), fill=(70, 150, 255, 110))
    g.ellipse((w - 260, h - 300, w + 80, h + 60), fill=(40, 120, 230, 120))
    img = Image.alpha_composite(img, glow.filter(ImageFilter.GaussianBlur(60)))

    rng = random.Random(7)
    bubbles = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(bubbles)
    for _ in range(16):
        r = rng.randint(20, 90)
        cx, cy = rng.randint(-10, w + 10), rng.randint(0, h)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(120, 190, 255, rng.randint(16, 36)),
                  outline=(200, 230, 255, rng.randint(50, 110)), width=2)
        d.ellipse((cx - r * 0.6, cy - r * 0.72, cx - r * 0.1, cy - r * 0.4),
                  fill=(255, 255, 255, rng.randint(40, 90)))
    return Image.alpha_composite(img, bubbles.filter(ImageFilter.GaussianBlur(1.2)))


def button_image(base, box, state):
    x, y, w, h = box
    styles = {"n": ((110, 185, 255), 125, 60), "h": ((150, 210, 255), 165, 95), "a": ((40, 120, 240), 230, 150)}
    tint, top, bottom = styles[state]
    out = Image.alpha_composite(base.crop((x, y, x + w, y + h)), glass(base, box, h // 2, tint, top, bottom, 6))
    return out.convert("RGB")


def make_knob(size=30):
    ss = 4
    S = size * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((ss * 2, ss * 2, S - ss * 2, S - ss * 2), fill=(150, 205, 255, 235),
              outline=(255, 255, 255, 255), width=ss)
    d.ellipse((S * 0.24, S * 0.17, S * 0.62, S * 0.45), fill=(255, 255, 255, 170))
    return img.resize((size, size), Image.LANCZOS)


class GlassButton:
    def __init__(self, canvas, base, box, text, command):
        x, y, w, h = box
        self.canvas, self.command, self.on = canvas, command, False
        self.imgs = {s: ImageTk.PhotoImage(button_image(base, box, s)) for s in ("n", "h", "a")}
        self.image = canvas.create_image(x, y, anchor="nw", image=self.imgs["n"])
        self.label = canvas.create_text(x + w / 2, y + h / 2, text=text, fill="#f4faff",
                                        font=("Segoe UI", 11, "bold"))
        for item in (self.image, self.label):
            canvas.tag_bind(item, "<Enter>", self._enter)
            canvas.tag_bind(item, "<Leave>", self._leave)
            canvas.tag_bind(item, "<ButtonPress-1>", lambda e: self._set("a"))
            canvas.tag_bind(item, "<ButtonRelease-1>", self._click)

    def _set(self, state):
        self.canvas.itemconfig(self.image, image=self.imgs[state])

    def _enter(self, e):
        self.canvas.configure(cursor="hand2")
        self._set("a" if self.on else "h")

    def _leave(self, e):
        self.canvas.configure(cursor="")
        self._set("a" if self.on else "n")

    def _click(self, e):
        self.command()
        self._set("a" if self.on else "h")

    def set_on(self, value, text):
        self.on = value
        self.canvas.itemconfig(self.label, text=text)
        self._set("a" if value else "n")


class GlassSlider:
    def __init__(self, canvas, knob, x, y, w, lo, hi, step, value, title, fmt, on_change):
        self.canvas, self.knob_img = canvas, knob
        self.x, self.y, self.w, self.lo, self.hi, self.step = x, y, w, lo, hi, step
        self.fmt, self.on_change = fmt, on_change
        canvas.create_text(x, y - 24, anchor="w", text=title, fill="#eaf6ff", font=("Segoe UI", 11, "bold"))
        self.value_text = canvas.create_text(x + w, y - 24, anchor="e", fill="#bfe1ff", font=("Segoe UI", 10))
        self.track = canvas.create_line(x, y, x + w, y, width=9, capstyle="round", fill="#0a2d62")
        self.fill = canvas.create_line(x, y, x + 1, y, width=9, capstyle="round", fill="#5cb8ff")
        self.handle = canvas.create_image(x, y, image=knob)
        for item in (self.track, self.fill, self.handle):
            canvas.tag_bind(item, "<ButtonPress-1>", self._drag)
            canvas.tag_bind(item, "<B1-Motion>", self._drag)
        self.set(value)

    def set(self, value):
        self.value = value
        frac = (value - self.lo) / (self.hi - self.lo)
        px = self.x + self.w * frac
        self.canvas.coords(self.fill, self.x, self.y, max(px, self.x + 1), self.y)
        self.canvas.coords(self.handle, px, self.y)
        self.canvas.itemconfig(self.value_text, text=self.fmt(value))

    def _drag(self, e):
        frac = min(1.0, max(0.0, (e.x - self.x) / self.w))
        value = round((self.lo + frac * (self.hi - self.lo)) / self.step) * self.step
        value = int(min(self.hi, max(self.lo, value)))
        self.set(value)
        self.on_change(value)


class SpawnerUI:
    def __init__(self, root):
        self.root = root
        self.spawned = []
        self.delete_mode = False
        self.last_shed = 0.0
        self._build()
        self._govern()

    def _build(self):
        root = self.root
        root.title("Character Spawner")
        root.configure(bg="#061a3a")
        root.resizable(False, False)
        root.geometry(f"{WIN_W}x{WIN_H}+{(root.winfo_screenwidth() - WIN_W) // 2}+"
                      f"{(root.winfo_screenheight() - WIN_H) // 2}")

        base = make_background(WIN_W, WIN_H)
        for x, y, w, h, r in ((24, 90, 512, 62, 26), (24, 234, 512, 290, 30)):
            base.alpha_composite(glass(base, (x, y, w, h), r, (150, 200, 255), 64, 22, 22), (x, y))
        self.bg_photo = ImageTk.PhotoImage(base.convert("RGB"))
        self.canvas = tk.Canvas(root, width=WIN_W, height=WIN_H, highlightthickness=0, bd=0)
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.bg_photo)

        self.canvas.create_text(WIN_W / 2, 36, text="Character Spawner", fill="#f2f9ff",
                                font=("Segoe UI", 21, "bold"))
        self.canvas.create_text(WIN_W / 2, 66, text="Image  GIF  APNG  Video  YouTube  3D Model",
                                fill="#9cc7ff", font=("Segoe UI", 9))

        self.entry = tk.Entry(root, font=("Segoe UI", 11), bg="#123a74", fg="#eef7ff",
                              insertbackground="#ffffff", relief="flat", bd=0, highlightthickness=0)
        self.entry.bind("<Return>", lambda e: self._spawn())
        self.canvas.create_window(44, 106, anchor="nw", width=336, height=30, window=self.entry)

        self.buttons = [
            GlassButton(self.canvas, base, (396, 100, 124, 42), "Browse", self._browse),
            GlassButton(self.canvas, base, (24, 168, 160, 46), "Spawn", self._spawn),
        ]
        self.delete_button = GlassButton(self.canvas, base, (200, 168, 160, 46), "Delete Mode",
                                         self._toggle_delete)
        self.clear_button = GlassButton(self.canvas, base, (376, 168, 160, 46), "Clear All", self._clear_all)

        self.canvas.create_text(44, 262, anchor="w", text="Resource Limits", fill="#f2f9ff",
                                font=("Segoe UI", 13, "bold"))
        knob = make_knob()
        self.knob = knob_photo = ImageTk.PhotoImage(knob)
        total = int(psutil.virtual_memory().total / 1048576)
        ram_max = max(1024, min(total, 16384) // 128 * 128)
        GOV.ram = min(GOV.ram, ram_max)
        self.sliders = [
            GlassSlider(self.canvas, knob_photo, 44, 330, 472, 5, 100, 5, GOV.cpu, "CPU max",
                        lambda v: f"{v}%", lambda v: setattr(GOV, "cpu", v)),
            GlassSlider(self.canvas, knob_photo, 44, 404, 472, 256, ram_max, 128, GOV.ram, "RAM max",
                        lambda v: f"{v} MB", lambda v: setattr(GOV, "ram", v)),
            GlassSlider(self.canvas, knob_photo, 44, 478, 472, 10, 100, 5, GOV.gpu, "GPU max (FPS and detail)",
                        lambda v: f"{v}%  /  {max(5, int(60 * v / 100))} fps", lambda v: setattr(GOV, "gpu", v)),
        ]
        self.usage = self.canvas.create_text(WIN_W / 2, 508, fill="#9cc7ff", font=("Segoe UI", 9))
        self.counter = self.canvas.create_text(30, 558, anchor="w", text="Spawned: 0",
                                               fill="#bfe1ff", font=("Segoe UI", 10))
        self.status = self.canvas.create_text(WIN_W / 2, 586, fill="#ffe08a", font=("Segoe UI", 9))
        root.protocol("WM_DELETE_WINDOW", root.destroy)

    def _status(self, msg):
        self.root.after(0, lambda: self.canvas.itemconfig(self.status, text=msg))

    def _tick(self):
        self.canvas.itemconfig(self.counter, text=f"Spawned: {len(self.spawned)}")

    def _browse(self):
        path = filedialog.askopenfilename(title="Select File", filetypes=FILE_TYPES, initialdir=start_dir())
        if path:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, path)

    def _spawn(self):
        path = self.entry.get().strip()
        if not path:
            self._browse()
            path = self.entry.get().strip()
        if not path:
            return
        if GOV.over_ram():
            self._status(f"RAM limit reached ({GOV.rss:.0f} / {GOV.ram} MB)")
            return
        if is_youtube_url(path):
            self._status("Loading YouTube stream...")
            threading.Thread(target=self._youtube, args=(path,), daemon=True).start()
            return
        if not is_url(path) and not os.path.isfile(path):
            messagebox.showwarning("ไม่พบไฟล์", f"ไม่พบไฟล์:\n{path}")
            return
        self._add(path)

    def _youtube(self, url):
        if not ensure("yt_dlp", "yt-dlp"):
            self.root.after(0, lambda: messagebox.showerror("Error", "ติดตั้ง yt-dlp ไม่สำเร็จ\nรัน: pip install yt-dlp"))
            self._status("")
            return
        import yt_dlp
        options = {"format": "best[ext=mp4]/best", "quiet": True, "no_warnings": True, "noplaylist": True}
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=False)
                stream = info.get("url") or info["formats"][-1]["url"]
        except Exception as e:
            self.root.after(0, lambda m=str(e): messagebox.showerror("YouTube Error", m))
            self._status("")
            return
        self._status("")
        self.root.after(0, lambda: self._add(stream))

    def _add(self, path):
        try:
            source = make_source(path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self.spawned.append(CharacterWindow(self.root, self, source, path, on_close=self._closed))
        self._tick()

    def _closed(self, cw):
        if cw in self.spawned:
            self.spawned.remove(cw)
        self._tick()

    def _toggle_delete(self):
        self.delete_mode = not self.delete_mode
        self.delete_button.set_on(self.delete_mode, "Delete Mode: ON" if self.delete_mode else "Delete Mode")
        for cw in self.spawned:
            cw.refresh_cursor()
        self._status("Click a character to delete it" if self.delete_mode else "")

    def _clear_all(self):
        for cw in self.spawned[:]:
            cw.close()
        self._tick()

    def _govern(self):
        GOV.poll()
        if GOV.rss > GOV.ram:
            gc.collect()
            if GOV.rss > GOV.ram * 1.1 and self.spawned and time.time() - self.last_shed > 5:
                self.spawned[-1].close()
                self.last_shed = time.time()
                self._status("RAM limit: newest character removed")
        self.canvas.itemconfig(self.usage, text=f"Now   CPU {GOV.cpu_now:.0f}%   RAM {GOV.rss:.0f} MB")
        self.root.after(500, self._govern)


def main():
    root = tk.Tk()
    SpawnerUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
