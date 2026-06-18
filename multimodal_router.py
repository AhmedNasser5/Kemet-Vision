"""
Multimodal Router — EfficientNet-B4  (v1)
==========================================
Handles:
- Image → OOD check → classification (EfficientNet-B4)
- Image → RAG query generation
- Smart entity tracking
- Reset signal for KemetCore

Architecture changes vs old router (EfficientNet-B2):
- Backbone  : EfficientNet-B4 (1792-dim embeddings, IMG_SIZE=380)
- Head      : 1792 → 1024 → 512 → num_classes  (deeper than B2's 1408 → 512 → 256)
- Classes   : 74 (was 28)
- Checkpoint: efficientnet_b4_complete_final.pth

OOD Thresholds:
- STRICT  0.50  → web/clean images    (unchanged — safe)
- CAMERA  0.44  → real-world photos   (unchanged — calibrated)

Similarity % shown in every result so the caller can log/display it.
"""

import io
import os
import re
import uuid
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageOps, ImageEnhance
import numpy as np

# =========================================================
# CONFIG
# =========================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"]  = ""

MODEL_PATH = "efficientnet_b4_complete_final.pth"
DEVICE     = torch.device("cpu")

# ── OOD thresholds ────────────────────────────────────────────────────────────
# These are applied to the MAX cosine similarity between the query image
# embedding and the full training-set embedding matrix (N_train × 1792).
#
# How we chose 0.50 / 0.44:
#   • Training embeddings are L2-normalised → cosine similarity ∈ [-1, 1].
#   • On the validation split the 5th-percentile similarity is saved in the
#     checkpoint as `similarity_threshold` (typically ≈ 0.72–0.85 for B4).
#   • 0.50 (STRICT) is intentionally conservative: well below the training
#     distribution so real artifacts always pass, while completely unrelated
#     images (noise, faces, landscapes) are rejected. ~95 % of museum images
#     score above 0.65, so 0.50 gives a comfortable safety margin.
#   • 0.44 (CAMERA) is slightly relaxed for phone photos because handheld
#     shooting, glass reflections, and JPEG compression can push similarity
#     down by 5–10 points compared to a clean scan of the same object.
#     Even at 0.44 true OOD images (e.g. food photos, selfies) are still
#     rejected — they typically score below 0.30.
#
# You can override both values at runtime via environment variables:
#   KEMET_THRESHOLD_STRICT=0.55  python -m uvicorn ...
OOD_THRESHOLD_STRICT = float(os.getenv("KEMET_THRESHOLD_STRICT", "0.50"))
OOD_THRESHOLD_CAMERA = float(os.getenv("KEMET_THRESHOLD_CAMERA", "0.44"))


# =========================================================
# MODEL ARCHITECTURE  (must match training notebook exactly)
# =========================================================

def _build_efficientnet_b4(num_classes: int) -> nn.Module:
    """
    EfficientNet-B4 with the custom 3-layer head used in training.
    Input resolution: 380 × 380 (native B4 resolution — preserves pretrained features).
    """
    model = models.efficientnet_b4(weights=None)
    in_f  = model.classifier[1].in_features   # 1792

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.50, inplace=True),
        nn.Linear(in_f, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),
        nn.Dropout(p=0.40),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.30),
        nn.Linear(512, num_classes),
    )
    return model


class _EfficientNetB4Encoder(nn.Module):
    """
    Backbone-only wrapper — strips the classifier head.
    Returns a pooled 1792-dim feature vector used exclusively for
    the cosine-similarity OOD filter.
    """
    def __init__(self, full_model: nn.Module):
        super().__init__()
        self.features = full_model.features
        self.avgpool  = full_model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return x.flatten(1)   # (batch, 1792)


# =========================================================
# LOAD CHECKPOINT
# =========================================================

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

_required = [
    "model_state_dict", "num_classes", "class_names",
    "img_size", "similarity_threshold", "train_embeddings",
    "normalize_mean", "normalize_std",
]
_missing = [k for k in _required if k not in _ckpt]
if _missing:
    raise KeyError(
        f"Checkpoint missing keys: {_missing}\n"
        "Re-run Cell 22 (SAVE COMPLETE CHECKPOINT) in the training notebook."
    )

num_classes : int  = _ckpt["num_classes"]          # 74
class_names : list = _ckpt["class_names"]
_img_size   : int  = _ckpt["img_size"]             # 380
_mean              = _ckpt["normalize_mean"]        # ImageNet mean
_std               = _ckpt["normalize_std"]         # ImageNet std

# Training-set embedding matrix — shape (N_train, 1792), L2-normalised
_train_embeddings: torch.Tensor = _ckpt["train_embeddings"].float().cpu()

# Checkpoint-stored threshold (5th percentile of validation similarities).
# We expose this for logging but use our own STRICT/CAMERA values at runtime
# so the caller can tune without re-saving the checkpoint.
_ckpt_threshold: float = float(_ckpt["similarity_threshold"])

_model = _build_efficientnet_b4(num_classes)
_model.load_state_dict(_ckpt["model_state_dict"])
_model.eval()

_encoder = _EfficientNetB4Encoder(_model).eval()

print(f"[Router] Model             — EfficientNet-B4")
print(f"[Router] Classes           — {num_classes}")
print(f"[Router] Image size        — {_img_size}×{_img_size}")
print(f"[Router] Embeddings        — {_train_embeddings.shape}")
print(f"[Router] Checkpoint thresh — {_ckpt_threshold:.4f}  (5th-pct of val set)")
print(f"[Router] Runtime STRICT    — {OOD_THRESHOLD_STRICT}  (web/clean images)")
print(f"[Router] Runtime CAMERA    — {OOD_THRESHOLD_CAMERA}  (real-world photos)")
print(f"[Router] Class list        — {class_names}")


# =========================================================
# IMAGE TRANSFORM
# =========================================================

_transform = transforms.Compose([
    transforms.Resize(
        (_img_size, _img_size),
        interpolation=transforms.InterpolationMode.BICUBIC,
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=_mean, std=_std),
])


# =========================================================
# IMAGE LOADING  (handles all input types + EXIF fix)
# =========================================================

def _load_pil_image(file_input) -> Image.Image:
    """
    Safely load a PIL image from:
      - bytes / bytearray      (camera capture from mobile app)
      - file-like object       (UploadFile, BytesIO, open file)
      - file path string

    Also corrects EXIF rotation — critical for phone/camera photos.

    Raises:
        ValueError if the image cannot be read.
    """
    try:
        if isinstance(file_input, (bytes, bytearray)):
            buffer = io.BytesIO(file_input)

        elif isinstance(file_input, str):
            buffer = open(file_input, "rb")

        elif hasattr(file_input, "read"):
            raw    = file_input.read()
            buffer = io.BytesIO(raw)

        else:
            raise TypeError(f"Unsupported input type: {type(file_input)}")

        with Image.open(buffer) as img:
            img = ImageOps.exif_transpose(img)   # fix phone rotation
            return img.convert("RGB")

    except (TypeError, ValueError):
        raise
    except Exception as exc:
        raise ValueError(f"Cannot read image: {exc}") from exc


# =========================================================
# CAMERA DETECTION
# =========================================================

def _is_camera_photo(pil_img: Image.Image) -> bool:
    """
    Heuristic: is this a camera/real-world photo or a clean web image?

    Two independent signals:
      1. EXIF metadata — phones always write EXIF; web images usually don't.
      2. Local noise variance in 8×8 patches — camera sensor noise is
         measurably higher than processed/compressed web images in smooth areas.

    Returns:
        True  → camera photo → use OOD_THRESHOLD_CAMERA (0.44)
        False → web/clean   → use OOD_THRESHOLD_STRICT  (0.50)
    """
    # Signal 1: EXIF
    try:
        exif     = pil_img._getexif()
        has_exif = exif is not None and len(exif) > 2
    except Exception:
        has_exif = False

    if has_exif:
        return True

    # Signal 2: local noise variance
    try:
        small      = pil_img.resize((64, 64))
        arr        = np.array(small, dtype=np.float32)
        patches    = arr.reshape(8, 8, 8, 8, 3)
        local_vars = patches.var(axis=(2, 3)).mean()
        if float(local_vars) > 35.0:
            return True
    except Exception:
        pass

    return False


# =========================================================
# CAMERA IMAGE PREPROCESSING
# =========================================================

def _preprocess_camera_image(pil_img: Image.Image) -> Image.Image:
    """
    Light enhancement for camera photos ONLY — never applied to web images.

    - Contrast  ×1.25 → handles glare / uneven museum lighting
    - Sharpness ×1.30 → handles motion blur from handheld shooting
    - Color     ×1.15 → handles washed-out colors from glass reflections

    Conservative values — helps similarity without distorting learned features.
    """
    img = ImageEnhance.Contrast(pil_img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.30)
    img = ImageEnhance.Color(img).enhance(1.15)
    return img


# =========================================================
# OOD SIMILARITY  (core computation)
# =========================================================

@torch.no_grad()
def _compute_similarity(tensor: torch.Tensor) -> float:
    """
    Max cosine similarity between image embedding and training-set matrix.

    Steps:
      1. Run tensor through backbone → 1792-dim feature vector.
      2. L2-normalise → unit vector on hypersphere.
      3. Dot product with L2-normalised training matrix → (1, N_train) cosine sims.
      4. Return the maximum — if ANY training image is close, the query passes.

    The result is always in [-1, 1]; for in-domain images it is typically
    above 0.65 after fine-tuning on Egyptian artifacts.
    """
    emb = F.normalize(_encoder(tensor), dim=1)          # (1, 1792)
    sim = (emb @ _train_embeddings.T).max().item()      # scalar
    return sim


# =========================================================
# ADAPTIVE OOD CHECK
# =========================================================

def _ood_check(
    pil_img: Image.Image,
) -> tuple[bool, float, str, float]:
    """
    Adaptive OOD check with dual-pass for camera photos.

    Camera photos:
      - Pass 1: original image
      - Pass 2: lightly preprocessed image (contrast/sharpness/color enhanced)
      - best_sim = max(pass1, pass2)
      - threshold = OOD_THRESHOLD_CAMERA

    Web/clean images:
      - Single pass — no preprocessing
      - threshold = OOD_THRESHOLD_STRICT

    Returns:
        (in_domain, best_similarity, source_label, threshold_used)
    """
    is_camera = _is_camera_photo(pil_img)

    if is_camera:
        threshold = OOD_THRESHOLD_CAMERA
        source    = "camera"

        sim_orig = _compute_similarity(_transform(pil_img).unsqueeze(0))
        pil_enh  = _preprocess_camera_image(pil_img)
        sim_enh  = _compute_similarity(_transform(pil_enh).unsqueeze(0))
        best_sim = max(sim_orig, sim_enh)

    else:
        threshold = OOD_THRESHOLD_STRICT
        source    = "web/clean"
        best_sim  = _compute_similarity(_transform(pil_img).unsqueeze(0))

    return best_sim >= threshold, best_sim, source, threshold


# =========================================================
# SESSION STORE
# =========================================================

class SessionStore:
    """Lightweight in-process session store for entity tracking."""

    def __init__(self):
        self.sessions: dict = {}

    def create_session(self, session_id=None) -> str:
        if not session_id:
            session_id = str(uuid.uuid4())[:8]
        if session_id not in self.sessions:
            self.sessions[session_id] = {"last_entity": None}
        return session_id

    def set_entity(self, session_id: str, entity: str):
        self.sessions[session_id]["last_entity"] = entity

    def get_entity(self, session_id: str):
        return self.sessions.get(session_id, {}).get("last_entity")

    def reset_entity(self, session_id: str):
        if session_id in self.sessions:
            self.sessions[session_id]["last_entity"] = None


session_store = SessionStore()


# =========================================================
# PUBLIC API: CLASSIFY WITH OOD FILTER
# =========================================================

@torch.no_grad()
def classify_with_ood_filter(
    file_input,
    top_k: int = 3,
) -> dict:
    """
    Full pipeline: load → detect source → adaptive OOD check → classify.

    Args:
        file_input : bytes, file-like object, or path string
        top_k      : number of top predictions to return (default 3)

    Returns dict with keys:
        in_domain           bool    — passed the OOD filter
        similarity          float   — max cosine sim to training set (0–1 scale)
        similarity_pct      str     — human-readable e.g. "78.3 %"
        image_source        str     — "camera" or "web/clean"
        threshold_used      float   — which threshold was applied
        classified          bool
        should_send_to_rag  bool
        predicted_class     str | None
        confidence          float | None
        top_k_predictions   list  | None
        message             str

    Raises:
        ValueError — if the image file cannot be read.
    """
    # ── Step 1: Load ──────────────────────────────────────────────────────────
    pil_img = _load_pil_image(file_input)

    # ── Step 2: Adaptive OOD check ────────────────────────────────────────────
    in_domain, sim, source, threshold_used = _ood_check(pil_img)

    sim_pct = f"{sim * 100:.1f} %"

    if not in_domain:
        return {
            "in_domain"         : False,
            "similarity"        : round(sim, 4),
            "similarity_pct"    : sim_pct,
            "image_source"      : source,
            "threshold_used"    : threshold_used,
            "classified"        : False,
            "should_send_to_rag": False,
            "predicted_class"   : None,
            "confidence"        : None,
            "top_k_predictions" : None,
            "message": (
                f"Image rejected — outside the known artifact domain. "
                f"Similarity: {sim:.4f} ({sim_pct}) < "
                f"threshold: {threshold_used:.4f}  "
                f"[source detected: {source}]"
            ),
        }

    # ── Step 3: Classify ──────────────────────────────────────────────────────
    # Use the original (non-enhanced) image for classification — the model was
    # trained on standard augmentations, not our enhancement pipeline.
    tensor             = _transform(pil_img).unsqueeze(0)
    probs              = F.softmax(_model(tensor), dim=1)
    conf               = probs.max().item()
    pred               = probs.argmax().item()
    top_probs, top_idx = probs[0].topk(min(top_k, num_classes))

    return {
        "in_domain"         : True,
        "similarity"        : round(sim, 4),
        "similarity_pct"    : sim_pct,
        "image_source"      : source,
        "threshold_used"    : threshold_used,
        "classified"        : True,
        "should_send_to_rag": True,
        "predicted_class"   : class_names[pred],
        "confidence"        : round(conf, 4),
        "top_k_predictions" : [
            {"class": class_names[i.item()], "confidence": round(p.item(), 4)}
            for p, i in zip(top_probs, top_idx)
        ],
        "message": (
            f"Identified: {class_names[pred]} "
            f"(confidence: {conf:.1%}, similarity: {sim_pct})"
        ),
    }


# =========================================================
# INTERNAL HELPER
# =========================================================

def extract_entity_from_image(file_input) -> str:
    """
    Returns the predicted class name, or raises ValueError if OOD/unreadable.
    Used internally by image_to_query().
    """
    result = classify_with_ood_filter(file_input)
    if not result["in_domain"]:
        raise ValueError(result["message"])
    return result["predicted_class"]


# =========================================================
# PUBLIC API: IMAGE → RAG QUERY
# =========================================================

def image_to_query(
    upload_file,
    session_id=None,
    user_question: str = None,
) -> tuple[str, str, bool]:
    """
    Full pipeline: OOD check → classify → build enriched RAG query.

    Args:
        upload_file   : image bytes / file-like / path
        session_id    : existing session ID (or None to create a new one)
        user_question : optional follow-up question from the user

    Returns:
        (final_query, session_id, reset_entity)

    Raises:
        ValueError — if image is OOD or unreadable.
    """
    session_id = session_store.create_session(session_id)
    entity     = extract_entity_from_image(upload_file)
    session_store.set_entity(session_id, entity)

    if user_question and user_question.strip():
        final_query = f"{user_question.strip()} about {entity}"
    else:
        final_query = (
            f"What is {entity}? Describe its historical significance, "
            f"origin, and role in ancient Egyptian culture."
        )

    return final_query, session_id, True


# =========================================================
# PUBLIC API: TEXT FOLLOW-UP → RAG QUERY
# =========================================================

def text_to_query(
    text: str,
    session_id=None,
) -> tuple[str, str, bool]:
    """
    Enrich a text follow-up with the session's last known entity.

    Detects new Named Entities in the text (capitalised phrases).
    If a new entity is found that differs from the session entity,
    the session is updated and reset_entity=True is returned so
    KemetCore can reset its context window.

    Returns:
        (final_query, session_id, reset_entity)
    """
    session_id   = session_store.create_session(session_id)
    entity       = session_store.get_entity(session_id)
    candidates   = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    new_entity   = candidates[0] if candidates else None
    reset_entity = False

    if new_entity and entity and new_entity != entity:
        # User switched topic — update session and signal reset
        session_store.set_entity(session_id, new_entity)
        reset_entity = True
        final_query  = text

    elif entity and not new_entity:
        # Continue talking about the same artifact
        final_query = f"{text} referring to {entity}"

    else:
        final_query = text

    return final_query, session_id, reset_entity
