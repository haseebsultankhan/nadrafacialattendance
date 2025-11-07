#!/usr/bin/env python3
"""
NADRA AI Attendance System - Flask Web Application
Multi-camera face recognition with role-based access control
"""

import os
import sys
import cv2
import time
import queue
import threading
import numpy as np
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from functools import wraps
import base64
import io
from PIL import Image

# ========================= 
# ======= CONFIG ========== 
# ========================= 

# Camera Configuration
CAMS = [
    ("192.168.1.206", "Main Entrance"),
    ("192.168.1.205", "Conference Hall"),
    ("192.168.1.203", "HOD SW/ML"),
    ("192.168.1.204", "IPC"),
]
USERNAME = "admin"
PASSWORD = "admin123"

# Model Paths
MODELS_DIR = "models"
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolo", "yolov12l-face.pt")
INSIGHTFACE_HOME = os.path.join(MODELS_DIR, "insightface_models")

# Detection Parameters
CONF_THRES = 0.45
IOU_THRES = 0.45
DETECT_GATE = 0.70
RECOG_SIM_THRESH = 0.30
EMB_DIM = 512
MAX_CANDIDATES = 1

# Performance Settings
BATCH_SIZE = 8
BATCH_TIMEOUT_MS = 15

# Databases
SUPERADMIN_DB_PATH = "superadmin.db"
ADMINS_DB_PATH = "admins.db"
EMPLOYEES_DB_PATH = "employees.db"

# Uploads
EMPLOYEES_FOLDER = "Employees"
DETECTION_IMAGES_FOLDER = "logs/detections"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp'}

# Flask
SECRET_KEY = "NADRA-AI-SECRET-KEY-CHANGE-THIS-IN-PRODUCTION"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EMPLOYEES_FOLDER, exist_ok=True)
os.makedirs(DETECTION_IMAGES_FOLDER, exist_ok=True)
os.makedirs(INSIGHTFACE_HOME, exist_ok=True)
os.environ.setdefault("INSIGHTFACE_HOME", INSIGHTFACE_HOME)

# ========================= 
# ====== DATABASE ========= 
# ========================= 

import sqlite3
from contextlib import contextmanager

@contextmanager
def get_superadmin_db():
    """Context manager for superadmin database connections."""
    conn = sqlite3.connect(SUPERADMIN_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def get_admins_db():
    """Context manager for admins database connections."""
    conn = sqlite3.connect(ADMINS_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

@contextmanager
def get_employees_db():
    """Context manager for employees database connections."""
    conn = sqlite3.connect(EMPLOYEES_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_databases():
    """Initialize all SQLite databases with required tables."""
    
    # SuperAdmin Database
    with get_superadmin_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS superadmins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        
        # Create default superadmins if not exist
        cursor.execute("SELECT COUNT(*) FROM superadmins")
        if cursor.fetchone()[0] == 0:
            superadmins = [
                ("haseeb.sultan", "Nadra@321"),
                ("m.saad", "Nadra@321"),
            ]
            for username, password in superadmins:
                password_hash = generate_password_hash(password)
                cursor.execute(
                    "INSERT INTO superadmins (username, password_hash) VALUES (?, ?)",
                    (username, password_hash)
                )
            conn.commit()
            print("[DB] Default superadmins created in superadmin.db")
    
    # Admins Database
    with get_admins_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                permissions TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT
            )
        """)
        conn.commit()
        print("[DB] Admins database initialized: admins.db")
    
    # Employees Database
    with get_employees_db() as conn:
        cursor = conn.cursor()
        
        # Employees table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emp_num INTEGER UNIQUE NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                department TEXT NOT NULL,
                photo_path TEXT NOT NULL,
                embedding BLOB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT
            )
        """)
        
        # Detections/Attendance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                emp_num INTEGER NOT NULL,
                full_name TEXT NOT NULL,
                camera_name TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                detection_image_path TEXT,
                similarity_score REAL,
                yolo_confidence REAL,
                FOREIGN KEY (emp_num) REFERENCES employees(emp_num)
            )
        """)
        
        # Audit log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                admin_username TEXT NOT NULL,
                action TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                details TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        print("[DB] Employees database initialized: employees.db")


def get_user_from_db(username):
    """Get user from appropriate database."""
    # Check superadmin database first
    with get_superadmin_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash FROM superadmins WHERE username = ?", (username,))
        user_data = cursor.fetchone()
        if user_data:
            return {
                'id': f"sa_{user_data[0]}",
                'username': user_data[1],
                'password_hash': user_data[2],
                'role': 'superadmin',
                'permissions': '{}'
            }
    
    # Check admins database
    with get_admins_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash, permissions FROM admins WHERE username = ?", (username,))
        user_data = cursor.fetchone()
        if user_data:
            return {
                'id': f"ad_{user_data[0]}",
                'username': user_data[1],
                'password_hash': user_data[2],
                'role': 'admin',
                'permissions': user_data[3] if user_data[3] else '{}'
            }
    
    return None


def scan_and_embed_employees(extractor):
    """Scan Employees folder and auto-embed existing photos."""
    if not os.path.exists(EMPLOYEES_FOLDER):
        print(f"[SCAN] Employees folder not found: {EMPLOYEES_FOLDER}")
        return
    
    with get_employees_db() as conn:
        cursor = conn.cursor()
        
        # Get existing employees
        cursor.execute("SELECT emp_num FROM employees")
        existing_emps = set(row[0] for row in cursor.fetchall())
        
        added_count = 0
        
        # Scan department folders
        for dept_name in os.listdir(EMPLOYEES_FOLDER):
            dept_path = os.path.join(EMPLOYEES_FOLDER, dept_name)
            
            if not os.path.isdir(dept_path):
                continue
            
            print(f"[SCAN] Scanning department: {dept_name}")
            
            # Scan employee photos
            for filename in os.listdir(dept_path):
                if not any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
                    continue
                
                try:
                    # Parse filename: {empnum}_{first}_{last}.jpg
                    name_part = os.path.splitext(filename)[0]
                    parts = name_part.split('_')
                    
                    if len(parts) < 3:
                        print(f"[SCAN] Invalid filename format: {filename}")
                        continue
                    
                    emp_num = int(parts[0])
                    first_name = parts[1]
                    last_name = '_'.join(parts[2:])
                    
                    # Skip if already exists
                    if emp_num in existing_emps:
                        continue
                    
                    # Load and process image
                    photo_path = os.path.join(dept_path, filename)
                    img = cv2.imread(photo_path)
                    
                    if img is None:
                        print(f"[SCAN] Failed to read: {photo_path}")
                        continue
                    
                    # Extract embedding
                    embeddings = extractor.embed_aligned_batch([img])
                    embedding = embeddings[0] if embeddings else None
                    
                    if embedding is None:
                        print(f"[SCAN] No face detected: {photo_path}")
                        continue
                    
                    # Save to database
                    embedding_blob = embedding.tobytes()
                    cursor.execute(
                        """INSERT INTO employees (emp_num, first_name, last_name, department, photo_path, embedding, created_by)
                        VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (emp_num, first_name, last_name, dept_name, photo_path, embedding_blob, "AUTO_SCAN")
                    )
                    conn.commit()
                    
                    print(f"[SCAN] ✅ Added: {emp_num} - {first_name} {last_name} ({dept_name})")
                    added_count += 1
                    existing_emps.add(emp_num)
                    
                except Exception as e:
                    print(f"[SCAN] Error processing {filename}: {e}")
                    continue
        
        print(f"[SCAN] Completed. Added {added_count} new employees")


def log_audit(admin_username: str, action: str, target_type: str, target_id: str, details: str = ""):
    """Log an admin action to audit log."""
    with get_employees_db() as conn:
        conn.execute(
            "INSERT INTO audit_log (admin_username, action, target_type, target_id, details) VALUES (?, ?, ?, ?, ?)",
            (admin_username, action, target_type, target_id, details)
        )
        conn.commit()


# ========================= 
# ===== FLASK SETUP ======= 
# ========================= 

app = Flask(__name__)
app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin):
    """User class for Flask-Login."""
    def __init__(self, id, username, role, permissions='{}'):
        self.id = id
        self.username = username
        self.role = role
        try:
            self.permissions = json.loads(permissions) if permissions else {}
        except:
            self.permissions = {}
    
    def is_superadmin(self):
        return self.role == 'superadmin'
    
    def has_permission(self, perm):
        if self.is_superadmin():
            return True
        return self.permissions.get(perm, False)


@login_manager.user_loader
def load_user(user_id):
    """Load user from appropriate database based on ID prefix."""
    if user_id.startswith('sa_'):
        # SuperAdmin
        actual_id = int(user_id.replace('sa_', ''))
        with get_superadmin_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username FROM superadmins WHERE id = ?", (actual_id,))
            user_data = cursor.fetchone()
            if user_data:
                return User(f"sa_{user_data[0]}", user_data[1], 'superadmin', '{}')
    
    elif user_id.startswith('ad_'):
        # Admin
        actual_id = int(user_id.replace('ad_', ''))
        with get_admins_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, username, permissions FROM admins WHERE id = ?", (actual_id,))
            user_data = cursor.fetchone()
            if user_data:
                return User(f"ad_{user_data[0]}", user_data[1], 'admin', user_data[2])
    
    return None


def permission_required(permission):
    """Decorator to require specific permission."""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not current_user.is_authenticated:
                return redirect(url_for('login'))
            if not current_user.has_permission(permission):
                flash(f"Access denied. {permission.replace('_', ' ').title()} permission required.", "danger")
                return redirect(url_for('dashboard'))
            return f(*args, **kwargs)
        return decorated_function
    return decorator


def superadmin_required(f):
    """Decorator to require superadmin role."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_superadmin():
            flash("Access denied. SuperAdmin privileges required.", "danger")
            return redirect(url_for('dashboard'))
        return f(*args, **kwargs)
    return decorated_function


# ========================= 
# ====== ML MODELS ======== 
# ========================= 

def select_device_str() -> Tuple[str, bool]:
    """Pick best device: CUDA → MPS → CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            return ("cuda:0", False)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return ("mps", False)
    except Exception:
        pass
    return ("cpu", False)


class YOLOFaceDetector:
    """Ultralytics YOLO face detector with batch support."""
    def __init__(self, model_path: str, conf_thres: float, iou_thres: float):
        from ultralytics import YOLO
        self.device, _ = select_device_str()
        print(f"[YOLO] Loading model: {model_path}")
        print(f"[YOLO] Device: {self.device}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.model.overrides["conf"] = conf_thres
        self.model.overrides["iou"] = iou_thres
        self.lock = threading.Lock()
    
    @staticmethod
    def _clip_xyxy(xyxy: np.ndarray, w: int, h: int) -> np.ndarray:
        xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, w - 1)
        xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, h - 1)
        return xyxy
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """Batch detection for multiple frames."""
        with self.lock:
            results = self.model.predict(
                source=frames,
                imgsz=480,
                device=self.device,
                verbose=False,
                stream=False
            )
        
        all_detections = []
        for idx, r in enumerate(results):
            frame_dets = []
            if r.boxes is not None and len(r.boxes) > 0:
                h, w = frames[idx].shape[:2]
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                xyxy = self._clip_xyxy(xyxy, w, h)
                
                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                    frame_dets.append({
                        "box": (int(x1), int(y1), int(x2), int(y2)),
                        "conf": float(confs[i])
                    })
            all_detections.append(frame_dets)
        
        return all_detections


class EmbeddingExtractor:
    """ArcFace embeddings via InsightFace."""
    def __init__(self):
        from insightface.app import FaceAnalysis
        
        if sys.platform == "darwin":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        print(f"[ArcFace] Loading InsightFace model...")
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers,
            allowed_modules=None
        )
        
        device, _ = select_device_str()
        ctx_id = 0 if device.startswith("cuda") else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(160, 160))
        self.lock = threading.Lock()
        print(f"[ArcFace] Model loaded successfully")
    
    @staticmethod
    def _best_face(faces):
        """Select the largest face by area."""
        if not faces:
            return None
        areas = [(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces]
        return faces[int(np.argmax(areas))]
    
    def embed_aligned_batch(self, aligned_faces: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Extract embeddings from face crops."""
        if not aligned_faces:
            return []
        
        with self.lock:
            embeddings = []
            for face_crop in aligned_faces:
                try:
                    faces = self.app.get(face_crop)
                    f = self._best_face(faces)
                    if f is None or f.normed_embedding is None:
                        embeddings.append(None)
                    else:
                        embeddings.append(f.normed_embedding.astype(np.float32))
                except Exception as e:
                    print(f"[ArcFace] Error: {e}")
                    embeddings.append(None)
            return embeddings


class FaceIndex:
    """FAISS cosine-similarity index."""
    def __init__(self, dim: int = EMB_DIM):
        import faiss
        self.faiss = faiss
        self.dim = dim
        self.index = self.faiss.IndexFlatIP(dim)
        self.emp_meta: List[Tuple[int, str]] = []
    
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v_copy = v.copy()
        n = np.linalg.norm(v_copy, axis=1, keepdims=True) + 1e-12
        return v_copy / n
    
    def add(self, embs: np.ndarray, emp_nums: List[int], full_names: List[str]):
        assert embs.shape[1] == self.dim
        embs_norm = self._normalize(embs.astype(np.float32))
        self.index.add(embs_norm)
        for en, fn in zip(emp_nums, full_names):
            self.emp_meta.append((en, fn))
    
    def search(self, emb: np.ndarray, topk: int = 1):
        emb = emb.reshape(1, -1).astype(np.float32)
        emb = self._normalize(emb)
        sims, idxs = self.index.search(emb, topk)
        return sims[0], idxs[0]
    
    def size(self) -> int:
        return self.index.ntotal
    
    def meta(self, idx: int) -> Tuple[int, str]:
        return self.emp_meta[idx]
    
    def rebuild_from_db(self, extractor: EmbeddingExtractor):
        """Rebuild index from database."""
        with get_employees_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT emp_num, first_name, last_name, embedding FROM employees")
            rows = cursor.fetchall()
        
        if not rows:
            print("[INDEX] No employees in database")
            return
        
        emp_nums = []
        full_names = []
        embs = []
        
        for row in rows:
            emp_num = row[0]
            first_name = row[1]
            last_name = row[2]
            embedding_blob = row[3]
            
            if embedding_blob:
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                emp_nums.append(emp_num)
                full_names.append(f"{first_name} {last_name}")
                embs.append(embedding)
        
        if embs:
            embs_np = np.vstack(embs).astype(np.float32)
            self.index = self.faiss.IndexFlatIP(self.dim)
            self.emp_meta = []
            self.add(embs_np, emp_nums, full_names)
            print(f"[INDEX] Rebuilt with {self.size()} employees")


# ========================= 
# === CAMERA PROCESSOR ==== 
# ========================= 

@dataclass
class DetectionRequest:
    cam_id: int
    frame_id: int
    frame: np.ndarray
    timestamp: float


@dataclass
class DetectionResult:
    cam_id: int
    frame_id: int
    detections: List[Dict]
    annotated_frame: Optional[np.ndarray] = None


class GPUBatchProcessor(threading.Thread):
    """Central GPU processor for batch detection and recognition."""
    
    def __init__(self, detector: YOLOFaceDetector, extractor: EmbeddingExtractor, db_index: FaceIndex, cam_names: Dict[int, str]):
        super().__init__(daemon=True)
        self.detector = detector
        self.extractor = extractor
        self.db_index = db_index
        self.cam_names = cam_names
        
        self.request_queue = queue.Queue(maxsize=100)
        self.result_queues: Dict[int, queue.Queue] = {}
        self.stop_event = threading.Event()
        self.logged_employees: Dict[int, Dict[int, float]] = {}
        self.lock = threading.Lock()
    
    def register_camera(self, cam_id: int) -> queue.Queue:
        result_queue = queue.Queue(maxsize=10)
        self.result_queues[cam_id] = result_queue
        self.logged_employees[cam_id] = {}
        return result_queue
    
    def submit_frame(self, request: DetectionRequest):
        try:
            self.request_queue.put_nowait(request)
        except queue.Full:
            pass
    
    def _should_log(self, cam_id: int, emp_num: int) -> bool:
        """Check if employee should be logged (avoid duplicates within 5 minutes)."""
        now = time.time()
        last_log = self.logged_employees[cam_id].get(emp_num, 0)
        if now - last_log > 300:
            self.logged_employees[cam_id][emp_num] = now
            return True
        return False
    
    def _log_detection(self, cam_id: int, emp_num: int, full_name: str, sim: float, yolo_conf: float, face_crop: np.ndarray):
        """Save detection to database with image."""
        cam_name = self.cam_names.get(cam_id, f"Camera {cam_id}")
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_filename = f"{emp_num}_{timestamp_str}.jpg"
        image_path = os.path.join(DETECTION_IMAGES_FOLDER, image_filename)
        cv2.imwrite(image_path, face_crop)
        
        with get_employees_db() as conn:
            conn.execute(
                """INSERT INTO detections 
                (emp_num, full_name, camera_name, detection_image_path, similarity_score, yolo_confidence)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (emp_num, full_name, cam_name, image_path, sim, yolo_conf)
            )
            conn.commit()
    
    def run(self):
        """Main batch processing loop."""
        print("[GPU] Batch processor started")
        
        while not self.stop_event.is_set():
            batch = []
            deadline = time.time() + (BATCH_TIMEOUT_MS / 1000.0)
            
            while len(batch) < BATCH_SIZE:
                timeout = max(0.001, deadline - time.time())
                try:
                    req = self.request_queue.get(timeout=timeout)
                    batch.append(req)
                except queue.Empty:
                    break
            
            if not batch:
                continue
            
            frames = [req.frame for req in batch]
            all_detections = self.detector.detect_batch(frames)
            
            crops_to_embed = []
            crop_metadata = []
            
            for batch_idx, (req, detections) in enumerate(zip(batch, all_detections)):
                for det_idx, det in enumerate(detections):
                    if det["conf"] >= DETECT_GATE:
                        x1, y1, x2, y2 = det["box"]
                        pad = 10
                        x1p = max(0, x1 - pad)
                        y1p = max(0, y1 - pad)
                        x2p = min(req.frame.shape[1] - 1, x2 + pad)
                        y2p = min(req.frame.shape[0] - 1, y2 + pad)
                        
                        crop = req.frame[y1p:y2p, x1p:x2p]
                        if crop.size > 0:
                            crops_to_embed.append(crop)
                            crop_metadata.append((batch_idx, det_idx, crop))
            
            embeddings = []
            if crops_to_embed:
                embeddings = self.extractor.embed_aligned_batch(crops_to_embed)
            
            for crop_idx, (batch_idx, det_idx, crop) in enumerate(crop_metadata):
                emb = embeddings[crop_idx] if crop_idx < len(embeddings) else None
                
                if emb is not None:
                    with self.lock:
                        if self.db_index.size() > 0:
                            sims, idxs = self.db_index.search(emb, topk=MAX_CANDIDATES)
                            sim = float(sims[0])
                            idx = int(idxs[0])
                            
                            if idx >= 0 and sim >= RECOG_SIM_THRESH:
                                emp_num, full_name = self.db_index.meta(idx)
                                det = all_detections[batch_idx][det_idx]
                                det["emp_num"] = emp_num
                                det["full_name"] = full_name
                                det["sim"] = sim
                                
                                if self._should_log(batch[batch_idx].cam_id, emp_num):
                                    self._log_detection(
                                        batch[batch_idx].cam_id,
                                        emp_num,
                                        full_name,
                                        sim,
                                        det["conf"],
                                        crop
                                    )
            
            for req, detections in zip(batch, all_detections):
                annotated = req.frame.copy()
                
                for det in detections:
                    x1, y1, x2, y2 = det["box"]
                    conf = det["conf"]
                    
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 3)
                    
                    label = f"Face {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), (0, 0, 0), -1)
                    cv2.putText(annotated, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if "emp_num" in det:
                        emp_label = f"{det['emp_num']} | {det['full_name']} ({det['sim']:.2f})"
                        (tw2, th2), _ = cv2.getTextSize(emp_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(annotated, (x1, y1 + 2), (x1 + tw2 + 4, y1 + th2 + 10), (0, 0, 0), -1)
                        cv2.putText(annotated, emp_label, (x1 + 2, y1 + th2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
                
                result = DetectionResult(
                    cam_id=req.cam_id,
                    frame_id=req.frame_id,
                    detections=detections,
                    annotated_frame=annotated
                )
                
                try:
                    self.result_queues[req.cam_id].put_nowait(result)
                except queue.Full:
                    pass
        
        print("[GPU] Batch processor stopped")
    
    def stop(self):
        self.stop_event.set()
    
    def rebuild_index(self):
        """Rebuild face index from database."""
        with self.lock:
            self.db_index.rebuild_from_db(self.extractor)


class CameraWorker(threading.Thread):
    """Camera capture thread."""
    
    def __init__(self, cam_id: int, ip: str, cam_name: str, gpu_processor: GPUBatchProcessor):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.ip = ip
        self.cam_name = cam_name
        self.gpu_processor = gpu_processor
        self.result_queue = gpu_processor.register_camera(cam_id)
        self.stop_event = threading.Event()
        self.cap = None
        self.frame_counter = 0
        self.latest_frame = None
        self.lock = threading.Lock()
    
    def _open_capture_gstreamer(self) -> bool:
        url = f"rtsp://{USERNAME}:{PASSWORD}@{self.ip}:554/cam/realmonitor?channel=1&subtype=1"
        gst_str = (
            f"rtspsrc location={url} latency=100 timeout=5000000 protocols=tcp ! "
            f"queue max-size-buffers=2 leaky=downstream ! "
            f"rtph264depay ! h264parse ! avdec_h264 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        
        try:
            cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                time.sleep(0.3)
                for _ in range(3):
                    ok, frame = cap.read()
                    if ok and frame is not None:
                        print(f"[CAM {self.cam_id}] ✅ Connected: {self.cam_name}")
                        self.cap = cap
                        return True
                    time.sleep(0.1)
            cap.release()
        except Exception as e:
            print(f"[CAM {self.cam_id}] GStreamer failed: {e}")
        return False
    
    def _open_capture_fallback(self) -> bool:
        url = f"rtsp://{USERNAME}:{PASSWORD}@{self.ip}:554/cam/realmonitor?channel=1&subtype=1"
        try:
            cap = cv2.VideoCapture(url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                time.sleep(0.3)
                ok, frame = cap.read()
                if ok and frame is not None:
                    print(f"[CAM {self.cam_id}] ✅ FFmpeg connected: {self.cam_name}")
                    self.cap = cap
                    return True
            cap.release()
        except Exception as e:
            print(f"[CAM {self.cam_id}] FFmpeg failed: {e}")
        return False
    
    def run(self):
        while not self.stop_event.is_set():
            if self.cap is None:
                if not self._open_capture_gstreamer():
                    self._open_capture_fallback()
                if self.cap is None:
                    time.sleep(2.0)
                    continue
            
            ok, frame = self.cap.read()
            if not ok or frame is None:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                time.sleep(2.0)
                continue
            
            self.frame_counter += 1
            req = DetectionRequest(
                cam_id=self.cam_id,
                frame_id=self.frame_counter,
                frame=frame.copy(),
                timestamp=time.time()
            )
            self.gpu_processor.submit_frame(req)
            
            try:
                while not self.result_queue.empty():
                    result = self.result_queue.get_nowait()
                    if result.annotated_frame is not None:
                        with self.lock:
                            self.latest_frame = result.annotated_frame
            except queue.Empty:
                pass
        
        if self.cap:
            self.cap.release()
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get latest annotated frame."""
        with self.lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def stop(self):
        self.stop_event.set()


# ========================= 
# ===== GLOBAL STATE ====== 
# ========================= 

detector = None
extractor = None
db_index = None
gpu_processor = None
camera_workers = []


def initialize_system():
    """Initialize ML models and camera workers."""
    global detector, extractor, db_index, gpu_processor, camera_workers
    
    print("=" * 60)
    print("NADRA AI Attendance System - Initializing...")
    print("=" * 60)
    
    if not os.path.isfile(YOLO_MODEL_PATH):
        raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")
    
    detector = YOLOFaceDetector(YOLO_MODEL_PATH, CONF_THRES, IOU_THRES)
    extractor = EmbeddingExtractor()
    db_index = FaceIndex(dim=EMB_DIM)
    
    # Scan and embed existing employees
    scan_and_embed_employees(extractor)
    
    # Rebuild index
    db_index.rebuild_from_db(extractor)
    
    cam_names = {i: name for i, (_, name) in enumerate(CAMS)}
    gpu_processor = GPUBatchProcessor(detector, extractor, db_index, cam_names)
    gpu_processor.start()
    
    for i, (ip, name) in enumerate(CAMS):
        worker = CameraWorker(i, ip, name, gpu_processor)
        camera_workers.append(worker)
        worker.start()
    
    print("[SYSTEM] ✅ Initialization complete")


# ========================= 
# ====== FLASK ROUTES ===== 
# ========================= 

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/portal')
def portal():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        
        user_data = get_user_from_db(username)
        
        if user_data and check_password_hash(user_data['password_hash'], password):
            user = User(user_data['id'], user_data['username'], user_data['role'], user_data['permissions'])
            login_user(user)
            flash(f"Welcome, {user.username}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid username or password", "danger")
    
    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully", "info")
    return redirect(url_for('login'))


@app.route('/dashboard')
@login_required
def dashboard():
    with get_employees_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM employees")
        total_employees = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM detections WHERE DATE(timestamp) = DATE('now')")
        today_detections = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT emp_num) FROM detections WHERE DATE(timestamp) = DATE('now')")
        unique_today = cursor.fetchone()[0]
    
    return render_template('dashboard.html', 
                         total_employees=total_employees,
                         today_detections=today_detections,
                         unique_today=unique_today)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_departments():
    """Get list of departments from Employees folder."""
    if not os.path.exists(EMPLOYEES_FOLDER):
        return []
    return [d for d in os.listdir(EMPLOYEES_FOLDER) if os.path.isdir(os.path.join(EMPLOYEES_FOLDER, d))]


@app.route('/add_employee', methods=['GET', 'POST'])
@login_required
@permission_required('add_employee')
def add_employee():
    if request.method == 'POST':
        emp_num = request.form.get('emp_num', '').strip()
        first_name = request.form.get('first_name', '').strip()
        last_name = request.form.get('last_name', '').strip()
        department = request.form.get('department', '').strip()
        
        try:
            emp_num = int(emp_num)
        except ValueError:
            flash("Employee number must be an integer", "danger")
            return redirect(url_for('add_employee'))
        
        if not all([emp_num, first_name, last_name, department]):
            flash("All fields are required", "danger")
            return redirect(url_for('add_employee'))
        
        if 'photo' not in request.files:
            flash("Photo is required", "danger")
            return redirect(url_for('add_employee'))
        
        file = request.files['photo']
        if file.filename == '' or not allowed_file(file.filename):
            flash("Invalid file", "danger")
            return redirect(url_for('add_employee'))
        
        with get_employees_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM employees WHERE emp_num = ?", (emp_num,))
            if cursor.fetchone()[0] > 0:
                flash(f"Employee {emp_num} already exists", "danger")
                return redirect(url_for('add_employee'))
        
        dept_path = os.path.join(EMPLOYEES_FOLDER, department)
        os.makedirs(dept_path, exist_ok=True)
        
        filename = secure_filename(f"{emp_num}_{first_name}_{last_name}.{file.filename.rsplit('.', 1)[1]}")
        filepath = os.path.join(dept_path, filename)
        file.save(filepath)
        
        img = cv2.imread(filepath)
        if img is None:
            os.remove(filepath)
            flash("Failed to read image", "danger")
            return redirect(url_for('add_employee'))
        
        embeddings = extractor.embed_aligned_batch([img])
        embedding = embeddings[0] if embeddings else None
        
        if embedding is None:
            os.remove(filepath)
            flash("No face detected in image", "danger")
            return redirect(url_for('add_employee'))
        
        embedding_blob = embedding.tobytes()
        with get_employees_db() as conn:
            conn.execute(
                """INSERT INTO employees (emp_num, first_name, last_name, department, photo_path, embedding, created_by)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (emp_num, first_name, last_name, department, filepath, embedding_blob, current_user.username)
            )
            conn.commit()
        
        log_audit(current_user.username, "ADD_EMPLOYEE", "EMPLOYEE", str(emp_num), 
                 f"Added {first_name} {last_name} - {department}")
        
        gpu_processor.rebuild_index()
        
        flash(f"Employee {emp_num} added successfully", "success")
        return redirect(url_for('add_employee'))
    
    departments = get_departments()
    return render_template('add_employee.html', departments=departments)


@app.route('/remove_employee', methods=['GET', 'POST'])
@login_required
@permission_required('remove_employee')
def remove_employee():
    if request.method == 'POST':
        emp_num = request.form.get('emp_num', '').strip()
        
        try:
            emp_num = int(emp_num)
        except ValueError:
            flash("Invalid employee number", "danger")
            return redirect(url_for('remove_employee'))
        
        with get_employees_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT first_name, last_name, photo_path FROM employees WHERE emp_num = ?", (emp_num,))
            emp = cursor.fetchone()
            
            if not emp:
                flash(f"Employee {emp_num} not found", "danger")
                return redirect(url_for('remove_employee'))
            
            if os.path.exists(emp[2]):
                os.remove(emp[2])
            
            cursor.execute("DELETE FROM employees WHERE emp_num = ?", (emp_num,))
            conn.commit()
        
        log_audit(current_user.username, "REMOVE_EMPLOYEE", "EMPLOYEE", str(emp_num),
                 f"Removed {emp[0]} {emp[1]}")
        
        gpu_processor.rebuild_index()
        
        flash(f"Employee {emp_num} removed successfully", "success")
        return redirect(url_for('remove_employee'))
    
    with get_employees_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT emp_num, first_name, last_name, department FROM employees ORDER BY emp_num")
        employees = cursor.fetchall()
    
    return render_template('remove_employee.html', employees=employees)


@app.route('/add_admin', methods=['GET', 'POST'])
@login_required
@superadmin_required
def add_admin():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        role = request.form.get('role', 'admin')
        
        if role not in ['admin', 'superadmin']:
            flash("Invalid role", "danger")
            return redirect(url_for('add_admin'))
        
        if not username or not password:
            flash("Username and password required", "danger")
            return redirect(url_for('add_admin'))
        
        password_hash = generate_password_hash(password)
        
        if role == 'superadmin':
            try:
                with get_superadmin_db() as conn:
                    conn.execute(
                        "INSERT INTO superadmins (username, password_hash) VALUES (?, ?)",
                        (username, password_hash)
                    )
                    conn.commit()
                log_audit(current_user.username, "ADD_SUPERADMIN", "USER", username, "Added SuperAdmin")
                flash(f"SuperAdmin {username} added successfully", "success")
            except sqlite3.IntegrityError:
                flash(f"Username {username} already exists", "danger")
        
        else:  # admin
            permissions = {
                'add_employee': 'perm_add_employee' in request.form,
                'remove_employee': 'perm_remove_employee' in request.form,
                'live_view': 'perm_live_view' in request.form,
                'live_logs': 'perm_live_logs' in request.form,
                'reset_password_self': True
            }
            
            try:
                with get_admins_db() as conn:
                    conn.execute(
                        "INSERT INTO admins (username, password_hash, permissions, created_by) VALUES (?, ?, ?, ?)",
                        (username, password_hash, json.dumps(permissions), current_user.username)
                    )
                    conn.commit()
                log_audit(current_user.username, "ADD_ADMIN", "USER", username, "Added Admin")
                flash(f"Admin {username} added successfully", "success")
            except sqlite3.IntegrityError:
                flash(f"Username {username} already exists", "danger")
        
        return redirect(url_for('add_admin'))
    
    return render_template('add_admin.html')


@app.route('/remove_admin', methods=['GET', 'POST'])
@login_required
@superadmin_required
def remove_admin():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        role = request.form.get('role', '').strip()
        
        if username == current_user.username:
            flash("Cannot remove yourself", "danger")
            return redirect(url_for('remove_admin'))
        
        if role == 'superadmin':
            with get_superadmin_db() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM superadmins WHERE username = ?", (username,))
                conn.commit()
            log_audit(current_user.username, "REMOVE_SUPERADMIN", "USER", username, "Removed SuperAdmin")
        else:
            with get_admins_db() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM admins WHERE username = ?", (username,))
                conn.commit()
            log_audit(current_user.username, "REMOVE_ADMIN", "USER", username, "Removed Admin")
        
        flash(f"User {username} removed successfully", "success")
        return redirect(url_for('remove_admin'))
    
    # Get all superadmins
    superadmins = []
    with get_superadmin_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM superadmins WHERE username != ? ORDER BY username", 
                      (current_user.username,))
        superadmins = [(row[0], 'superadmin') for row in cursor.fetchall()]
    
    # Get all admins
    admins = []
    with get_admins_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM admins ORDER BY username")
        admins = [(row[0], 'admin') for row in cursor.fetchall()]
    
    all_users = superadmins + admins
    
    return render_template('remove_admin.html', admins=all_users)


@app.route('/live_view')
@login_required
@permission_required('live_view')
def live_view():
    return render_template('live_view.html', cameras=CAMS)


@app.route('/video_feed/<int:cam_id>')
@login_required
@permission_required('live_view')
def video_feed(cam_id):
    def generate():
        if cam_id >= len(camera_workers):
            return
        
        worker = camera_workers[cam_id]
        while True:
            frame = worker.get_latest_frame()
            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/live_logs')
@login_required
@permission_required('live_logs')
def live_logs():
    page = request.args.get('page', 1, type=int)
    per_page = 50
    
    with get_employees_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, emp_num, full_name, camera_name, timestamp, detection_image_path, similarity_score, yolo_confidence
            FROM detections ORDER BY timestamp DESC LIMIT ? OFFSET ?""",
            (per_page, (page - 1) * per_page)
        )
        logs = cursor.fetchall()
        
        cursor.execute("SELECT COUNT(*) FROM detections")
        total = cursor.fetchone()[0]
    
    total_pages = (total + per_page - 1) // per_page
    
    return render_template('live_logs.html', logs=logs, page=page, total_pages=total_pages)


@app.route('/notifications')
@login_required
@superadmin_required
def notifications():
    page = request.args.get('page', 1, type=int)
    per_page = 50
    
    with get_employees_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT id, admin_username, action, target_type, target_id, details, timestamp
            FROM audit_log ORDER BY timestamp DESC LIMIT ? OFFSET ?""",
            (per_page, (page - 1) * per_page)
        )
        notifications = cursor.fetchall()
        
        cursor.execute("SELECT COUNT(*) FROM audit_log")
        total = cursor.fetchone()[0]
    
    total_pages = (total + per_page - 1) // per_page
    
    return render_template('notifications.html', notifications=notifications, page=page, total_pages=total_pages)


@app.route('/reset_password', methods=['GET', 'POST'])
@login_required
def reset_password():
    if request.method == 'POST':
        new_password = request.form.get('new_password', '')
        
        if not current_user.is_superadmin():
            if not current_user.has_permission('reset_password_self'):
                flash("Access denied", "danger")
                return redirect(url_for('dashboard'))
            
            username = current_user.username
        else:
            username = request.form.get('username', '').strip()
            role = request.form.get('role', '').strip()
            
            if not username:
                flash("Username required", "danger")
                return redirect(url_for('reset_password'))
        
        if not new_password:
            flash("New password required", "danger")
            return redirect(url_for('reset_password'))
        
        password_hash = generate_password_hash(new_password)
        
        if not current_user.is_superadmin():
            # Admin resetting own password
            actual_id = int(current_user.id.replace('ad_', ''))
            with get_admins_db() as conn:
                cursor = conn.cursor()
                cursor.execute("UPDATE admins SET password_hash = ? WHERE id = ?", (password_hash, actual_id))
                conn.commit()
            log_audit(current_user.username, "RESET_PASSWORD_SELF", "USER", username, "Self password reset")
            flash("Password reset successfully", "success")
        else:
            # SuperAdmin resetting any password
            role = request.form.get('role', '')
            if role == 'superadmin':
                with get_superadmin_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE superadmins SET password_hash = ? WHERE username = ?", (password_hash, username))
                    if cursor.rowcount == 0:
                        flash(f"User {username} not found", "danger")
                    else:
                        conn.commit()
                        log_audit(current_user.username, "RESET_PASSWORD", "SUPERADMIN", username, "Password reset")
                        flash(f"Password reset for {username}", "success")
            else:
                with get_admins_db() as conn:
                    cursor = conn.cursor()
                    cursor.execute("UPDATE admins SET password_hash = ? WHERE username = ?", (password_hash, username))
                    if cursor.rowcount == 0:
                        flash(f"User {username} not found", "danger")
                    else:
                        conn.commit()
                        log_audit(current_user.username, "RESET_PASSWORD", "ADMIN", username, "Password reset")
                        flash(f"Password reset for {username}", "success")
        
        return redirect(url_for('reset_password'))
    
    if not current_user.is_superadmin():
        return render_template('reset_password.html', users=None, self_only=True)
    
    # Get all superadmins
    superadmins = []
    with get_superadmin_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM superadmins ORDER BY username")
        superadmins = [(row[0], 'superadmin') for row in cursor.fetchall()]
    
    # Get all admins
    admins = []
    with get_admins_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM admins ORDER BY username")
        admins = [(row[0], 'admin') for row in cursor.fetchall()]
    
    all_users = superadmins + admins
    
    return render_template('reset_password.html', users=all_users, self_only=False)


@app.route('/detection_image/<int:detection_id>')
@login_required
def detection_image(detection_id):
    with get_employees_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT detection_image_path FROM detections WHERE id = ?", (detection_id,))
        result = cursor.fetchone()
    
    if result and os.path.exists(result[0]):
        with open(result[0], 'rb') as f:
            image_data = f.read()
        return Response(image_data, mimetype='image/jpeg')
    
    return "Image not found", 404


# ========================= 
# ======== MAIN =========== 
# ========================= 

if __name__ == '__main__':
    init_databases()
    initialize_system()
    
    print("\n" + "=" * 60)
    print("🚀 NADRA AI Attendance System")
    print("=" * 60)
    print("📡 Access the portal at: http://localhost:5000/portal")
    print("=" * 60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)