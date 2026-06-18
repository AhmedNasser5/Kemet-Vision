"""
app.py — Kemet Vision  v2.0
============================
EfficientNet-B4 + CLIP Ensemble + Cosine-Similarity OOD Filter
Egyptian Artifact Recognition System — 74 Classes

Run:
    streamlit run app.py
"""

# ── Fix OpenMP conflict BEFORE any other import ──────────────────────────────
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["CUDA_VISIBLE_DEVICES"]  = ""   # force CPU — avoids CUDA errors

import time
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import streamlit as st

DEVICE = torch.device("cpu")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kemet Vision · AI Artifact Recognition",
    page_icon="𓂀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT PATH
# ─────────────────────────────────────────────────────────────────────────────
CKPT_PATH = os.getenv(
    "KEMET_CKPT",
    r"C:\Ahmed Nasser\Artificial Intelligence\Graduation Project\CNN\Model5_EfficientNet-B4\Model 1\efficientnet_b4_complete_final.pth"
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Refined Luxury / Editorial Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;1,300;1,400&family=Josefin+Sans:wght@100;200;300;400;600&family=Playfair+Display:ital,wght@0,700;0,900;1,400&display=swap');

:root {
    --gold:         #BFA46A;
    --gold-light:   #D4BB84;
    --gold-bright:  #E8D4A0;
    --gold-dim:     #6B5A35;
    --gold-faint:   rgba(191,164,106,0.08);
    --obsidian:     #06050A;
    --ink:          #0B0A10;
    --dark1:        #0F0E15;
    --dark2:        #141320;
    --dark3:        #1A1928;
    --dark4:        #201F2E;
    --dark5:        #272535;
    --lapis:        #1A1F3A;
    --lapis-hi:     #252B4A;
    --cream:        #EDE6D6;
    --sand:         #9A8A70;
    --sand-light:   #B5A890;
    --border:       rgba(191,164,106,0.12);
    --border-hi:    rgba(191,164,106,0.30);
    --border-glow:  rgba(191,164,106,0.50);
    --green:        #2ECC71;
    --green-dim:    rgba(46,204,113,0.12);
    --red:          #E74C3C;
    --red-dim:      rgba(231,76,60,0.12);
    --amber:        #F39C12;
    --radius:       10px;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Cormorant Garamond', Georgia, serif !important;
    background-color: var(--obsidian) !important;
    color: var(--cream) !important;
}

#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

/* Custom scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--dark1); }
::-webkit-scrollbar-thumb { background: var(--gold-dim); border-radius: 2px; }

.block-container {
    padding: 0 2.5rem 4rem !important;
    max-width: 1440px !important;
}

/* ═══════════════════════════════════════
   HERO
═══════════════════════════════════════ */
.hero-wrap {
    position: relative;
    width: 100%;
    padding: 3.5rem 3rem 2.8rem;
    margin-bottom: 3rem;
    text-align: center;
    overflow: hidden;
}

.hero-bg {
    position: absolute;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(191,164,106,0.10) 0%, transparent 65%),
        radial-gradient(ellipse 40% 40% at 15% 50%, rgba(26,31,58,0.60) 0%, transparent 55%),
        radial-gradient(ellipse 40% 40% at 85% 50%, rgba(26,31,58,0.60) 0%, transparent 55%),
        linear-gradient(180deg, #0B0A10 0%, var(--obsidian) 100%);
    z-index: 0;
}

.hero-line-top {
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent 0%, var(--gold) 30%, var(--gold-bright) 50%, var(--gold) 70%, transparent 100%);
    z-index: 1;
}

.hero-line-bottom {
    position: absolute;
    bottom: 0; left: 10%; right: 10%;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-hi), transparent);
    z-index: 1;
}

.hero-content { position: relative; z-index: 2; }

.hero-hieroglyphs {
    font-size: 0.95rem;
    letter-spacing: 1.6rem;
    color: rgba(191,164,106,0.18);
    margin-bottom: 1.4rem;
    text-indent: 1.6rem;
}

.hero-eyebrow {
    font-family: 'Josefin Sans', sans-serif;
    font-weight: 200;
    font-size: 0.62rem;
    letter-spacing: 0.55em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 1rem;
}

.hero-title {
    font-family: 'Playfair Display', serif !important;
    font-weight: 900 !important;
    font-size: clamp(3rem, 6vw, 5.5rem) !important;
    line-height: 0.95 !important;
    background: linear-gradient(150deg,
        var(--gold-bright) 0%,
        var(--gold) 35%,
        var(--gold-dim) 60%,
        var(--gold) 80%,
        var(--gold-bright) 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    filter: drop-shadow(0 0 40px rgba(191,164,106,0.25)) !important;
    letter-spacing: 0.08em !important;
    margin: 0 !important;
}

.hero-title-sub {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-weight: 400;
    font-size: clamp(1rem, 2vw, 1.5rem);
    color: var(--sand);
    letter-spacing: 0.04em;
    margin: 0.6rem 0 1.8rem;
}

.hero-chips {
    display: flex;
    justify-content: center;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.hero-chip {
    font-family: 'Josefin Sans', sans-serif;
    font-weight: 300;
    font-size: 0.55rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--gold-dim);
    border: 1px solid var(--border);
    border-radius: 2px;
    padding: 0.28rem 0.75rem;
    background: rgba(191,164,106,0.04);
}

/* ═══════════════════════════════════════
   SECTION HEADINGS
═══════════════════════════════════════ */
.sec-label {
    font-family: 'Josefin Sans', sans-serif;
    font-weight: 300;
    font-size: 0.58rem;
    letter-spacing: 0.32em;
    text-transform: uppercase;
    color: var(--gold);
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.1rem;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-hi), transparent);
}

/* ═══════════════════════════════════════
   UPLOAD ZONE
═══════════════════════════════════════ */
[data-testid="stFileUploader"] > div {
    background: var(--gold-faint) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: var(--radius) !important;
    transition: all 0.3s ease !important;
    min-height: 160px !important;
}
[data-testid="stFileUploader"] > div:hover {
    background: rgba(191,164,106,0.10) !important;
    border-color: var(--gold) !important;
    box-shadow: 0 0 24px rgba(191,164,106,0.10) !important;
}
[data-testid="stFileUploader"] label { display: none !important; }

/* ═══════════════════════════════════════
   BUTTON
═══════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, var(--gold-dim) 0%, var(--gold) 50%, var(--gold-light) 100%) !important;
    color: var(--obsidian) !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Josefin Sans', sans-serif !important;
    font-size: 0.62rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.35em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 2rem !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(191,164,106,0.22) !important;
    margin-top: 0.8rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 36px rgba(191,164,106,0.35) !important;
    opacity: 0.92 !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ═══════════════════════════════════════
   RESULT CARDS
═══════════════════════════════════════ */
.card {
    border-radius: var(--radius);
    padding: 2.2rem 2.4rem 2rem;
    position: relative;
    overflow: hidden;
}

.card-known {
    background: linear-gradient(145deg, #060F09, #040B07);
    border: 1px solid rgba(46,204,113,0.25);
    box-shadow: 0 0 80px rgba(46,204,113,0.04), 0 20px 60px rgba(0,0,0,0.7);
}
.card-known::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(46,204,113,0.7), transparent);
}
.card-known::after {
    content: '𓂀';
    position: absolute;
    bottom: -0.5rem; right: 1.5rem;
    font-size: 7rem;
    color: rgba(46,204,113,0.04);
    line-height: 1;
    pointer-events: none;
}

.card-unknown {
    background: linear-gradient(145deg, #0F0606, #0A0404);
    border: 1px solid rgba(231,76,60,0.20);
    box-shadow: 0 0 80px rgba(231,76,60,0.04), 0 20px 60px rgba(0,0,0,0.7);
}
.card-unknown::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(231,76,60,0.5), transparent);
}

.card-verdict {
    font-family: 'Josefin Sans', sans-serif;
    font-weight: 300;
    font-size: 0.55rem;
    letter-spacing: 0.38em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}
.card-verdict-ok  { color: var(--green); }
.card-verdict-bad { color: var(--red); }

.card-name {
    font-family: 'Playfair Display', serif;
    font-weight: 700;
    font-size: clamp(1.4rem, 3vw, 2.2rem);
    color: var(--gold-light);
    line-height: 1.15;
    margin: 0 0 0.5rem;
}
.card-name-bad {
    font-family: 'Playfair Display', serif;
    font-weight: 400;
    font-style: italic;
    font-size: clamp(1.1rem, 2vw, 1.5rem);
    color: var(--red);
    line-height: 1.3;
    margin: 0 0 0.5rem;
}

.card-meta {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 0.95rem;
    color: var(--sand);
}

/* ═══════════════════════════════════════
   SIMILARITY METER
═══════════════════════════════════════ */
.meter-wrap { margin: 1.5rem 0 0.5rem; }
.meter-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.35rem;
}
.meter-label {
    font-family: 'Josefin Sans', sans-serif;
    font-weight: 300;
    font-size: 0.58rem;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    color: var(--sand);
}
.meter-value {
    font-family: 'Josefin Sans', sans-serif;
    font-weight: 400;
    font-size: 0.65rem;
    letter-spacing: 0.10em;
}
.meter-track {
    height: 5px;
    border-radius: 99px;
    background: rgba(255,255,255,0.04);
    overflow: hidden;
}
.meter-fill-ok {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, rgba(46,204,113,0.4), #2ecc71);
    box-shadow: 0 0 12px rgba(46,204,113,0.4);
}
.meter-fill-bad {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, rgba(231,76,60,0.4), #e74c3c);
    box-shadow: 0 0 12px rgba(231,76,60,0.4);
}

/* ═══════════════════════════════════════
   META CHIPS
═══════════════════════════════════════ */
.chip-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1rem 0; }
.chip {
    font-family: 'Josefin Sans', sans-serif;
    font-weight: 300;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--sand);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 0.25rem 0.7rem;
    background: rgba(255,255,255,0.02);
}
.chip b { color: var(--gold-light); font-weight: 400; }

/* ═══════════════════════════════════════
   TOP-K BARS
═══════════════════════════════════════ */
.bar-item { margin: 0.65rem 0; }
.bar-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 0.22rem;
}
.bar-class {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.92rem;
    color: var(--cream);
}
.bar-class-rank {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.52rem;
    font-weight: 300;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--sand);
    margin-right: 0.5rem;
}
.bar-conf {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.62rem;
    font-weight: 400;
    letter-spacing: 0.08em;
}
.bar-conf-hi  { color: #2ecc71; }
.bar-conf-mid { color: var(--amber); }
.bar-conf-lo  { color: var(--red); }
.bar-track {
    height: 4px;
    border-radius: 99px;
    background: rgba(255,255,255,0.04);
    overflow: hidden;
}
.bar-fill-gold { height: 100%; border-radius: 99px;
    background: linear-gradient(90deg, var(--gold-dim), var(--gold)); }
.bar-fill-dim  { height: 100%; border-radius: 99px;
    background: rgba(255,255,255,0.10); }

/* ═══════════════════════════════════════
   ENSEMBLE DETAILS TABLE
═══════════════════════════════════════ */
.ensemble-wrap {
    background: var(--dark2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    margin: 1rem 0;
}
.ensemble-row {
    display: flex;
    align-items: center;
    padding: 0.65rem 1.1rem;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    gap: 0.6rem;
}
.ensemble-row:last-child { border-bottom: none; }
.ensemble-row:first-child {
    background: rgba(191,164,106,0.05);
    border-bottom: 1px solid var(--border);
}
.ensemble-model {
    font-family: 'Josefin Sans', sans-serif;
    font-weight: 300;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--gold);
    flex: 0 0 140px;
}
.ensemble-pred {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.88rem;
    color: var(--cream);
    flex: 1;
}
.ensemble-conf {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.58rem;
    font-weight: 400;
    color: var(--sand);
    flex: 0 0 55px;
    text-align: right;
}
.agree-badge {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.agree-yes { background: var(--green); box-shadow: 0 0 6px rgba(46,204,113,0.6); }
.agree-no  { background: rgba(255,255,255,0.12); }

/* ═══════════════════════════════════════
   RAG STATUS
═══════════════════════════════════════ */
.rag-banner {
    border-radius: 6px;
    padding: 0.8rem 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-top: 1.2rem;
}
.rag-yes {
    background: var(--green-dim);
    border: 1px solid rgba(46,204,113,0.30);
}
.rag-no {
    background: rgba(243,156,18,0.08);
    border: 1px solid rgba(243,156,18,0.30);
}
.rag-icon { font-size: 1.1rem; }
.rag-text {
    font-family: 'Josefin Sans', sans-serif;
    font-weight: 300;
    font-size: 0.58rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}
.rag-text-yes { color: var(--green); }
.rag-text-no  { color: var(--amber); }

/* ═══════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background: var(--dark1) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.2rem !important;
}
.sidebar-title {
    font-family: 'Josefin Sans', sans-serif;
    font-weight: 200;
    font-size: 0.58rem;
    letter-spacing: 0.38em;
    text-transform: uppercase;
    color: var(--gold);
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 1.5rem;
}
.sidebar-heading {
    font-family: 'Josefin Sans', sans-serif;
    font-weight: 300;
    font-size: 0.55rem;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--gold);
    margin: 1.5rem 0 0.6rem;
}

div[data-testid="stSlider"] label {
    font-family: 'Josefin Sans', sans-serif !important;
    font-size: 0.58rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.20em !important;
    color: var(--gold) !important;
    text-transform: uppercase !important;
}
.stSpinner > div { border-top-color: var(--gold) !important; }
[data-testid="stImage"] img {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

.class-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.22rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.class-num {
    font-family: 'Josefin Sans', sans-serif;
    font-size: 0.50rem;
    font-weight: 200;
    color: var(--gold-dim);
    flex: 0 0 24px;
    letter-spacing: 0.05em;
}
.class-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 0.82rem;
    color: var(--sand);
}

.thresh-box {
    background: rgba(191,164,106,0.04);
    border: 1px solid var(--border);
    border-left: 2px solid var(--gold-dim);
    border-radius: 0 6px 6px 0;
    padding: 0.75rem 1rem;
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 0.82rem;
    color: rgba(210,190,155,0.50);
    margin-top: 0.6rem;
    line-height: 1.6;
}

/* ═══════════════════════════════════════
   EMPTY STATE
═══════════════════════════════════════ */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    text-align: center;
    background: var(--gold-faint);
}
.empty-glyph {
    font-size: 3rem;
    color: rgba(191,164,106,0.15);
    margin-bottom: 1rem;
}
.empty-text {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 0.95rem;
    color: var(--sand);
    opacity: 0.5;
}

/* ═══════════════════════════════════════
   FOOTER
═══════════════════════════════════════ */
.footer {
    text-align: center;
    padding: 2rem 0 1rem;
    border-top: 1px solid var(--border);
    margin-top: 3rem;
}
.footer-glyphs {
    font-size: 1rem;
    letter-spacing: 1.2rem;
    color: rgba(191,164,106,0.10);
    margin-bottom: 0.8rem;
    text-indent: 1.2rem;
}
.footer-text {
    font-family: 'Josefin Sans', sans-serif;
    font-weight: 200;
    font-size: 0.52rem;
    letter-spacing: 0.20em;
    text-transform: uppercase;
    color: rgba(139,115,85,0.30);
}

/* ═══════════════════════════════════════
   DIVIDER
═══════════════════════════════════════ */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-hi), transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE  — EfficientNet-B4 (must match training notebook exactly)
# ─────────────────────────────────────────────────────────────────────────────
def _build_model(num_classes: int) -> nn.Module:
    base = models.efficientnet_b4(weights=None)
    in_f = base.classifier[1].in_features          # 1792
    base.classifier = nn.Sequential(
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
    return base


class _Encoder(nn.Module):
    """Backbone-only — returns 1792-dim feature vector for OOD cosine similarity."""
    def __init__(self, full_model: nn.Module):
        super().__init__()
        self.features = full_model.features
        self.avgpool  = full_model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.avgpool(self.features(x)).flatten(1)   # (B, 1792)


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(path: str) -> dict:
    if not os.path.exists(path):
        st.error(
            f"**Checkpoint not found:** `{path}`\n\n"
            "Place **efficientnet_b4_complete_final.pth** next to app.py "
            "or set the `KEMET_CKPT` environment variable."
        )
        st.stop()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    required = [
        "model_state_dict", "num_classes", "class_names",
        "img_size", "similarity_threshold", "train_embeddings",
        "normalize_mean", "normalize_std",
    ]
    missing = [k for k in required if k not in ckpt]
    if missing:
        st.error(f"Checkpoint missing keys: `{missing}`")
        st.stop()

    mdl = _build_model(ckpt["num_classes"])
    mdl.load_state_dict(ckpt["model_state_dict"])
    mdl.eval()

    enc = _Encoder(mdl).eval()
    emb = ckpt["train_embeddings"].float().cpu()

    return {
        "model"      : mdl,
        "encoder"    : enc,
        "embeddings" : emb,
        "class_names": ckpt["class_names"],
        "img_size"   : ckpt["img_size"],
        "threshold"  : float(ckpt["similarity_threshold"]),
        "mean"       : ckpt["normalize_mean"],
        "std"        : ckpt["normalize_std"],
        "num_classes": ckpt["num_classes"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict(pil_img: Image.Image, md: dict, threshold: float, top_k: int = 5) -> dict:
    tfm = transforms.Compose([
        transforms.Resize(
            (md["img_size"], md["img_size"]),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(md["mean"], md["std"]),
    ])
    tensor = tfm(pil_img.convert("RGB")).unsqueeze(0)

    # OOD similarity check
    emb = F.normalize(md["encoder"](tensor), dim=1)
    sim = (emb @ md["embeddings"].T).max().item()

    if sim < threshold:
        return {
            "in_domain" : False,
            "similarity": round(sim, 4),
            "threshold" : round(threshold, 4),
        }

    # Classification
    probs           = F.softmax(md["model"](tensor), dim=1)[0]
    top_p, top_i    = probs.topk(min(top_k, md["num_classes"]))

    return {
        "in_domain"  : True,
        "similarity" : round(sim, 4),
        "threshold"  : round(threshold, 4),
        "predicted"  : md["class_names"][top_i[0].item()],
        "confidence" : round(top_p[0].item(), 4),
        "topk"       : [
            {"class": md["class_names"][i.item()], "confidence": round(p.item(), 4)}
            for p, i in zip(top_p, top_i)
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-bg"></div>
    <div class="hero-line-top"></div>
    <div class="hero-line-bottom"></div>
    <div class="hero-content">
        <div class="hero-hieroglyphs">𓀀 𓁀 𓂀 𓃀 𓄀 𓅀 𓆀 𓇀 𓈀 𓉀 𓊀</div>
        <div class="hero-eyebrow">Ancient Egyptian Artifact Intelligence</div>
        <div class="hero-title">Kemet Vision</div>
        <div class="hero-title-sub">The Oracle of Ancient Forms</div>
        <div class="hero-chips">
            <span class="hero-chip">EfficientNet-B4</span>
            <span class="hero-chip">CLIP ViT-L/14 · B/32 · B/16 Ensemble</span>
            <span class="hero-chip">74 Artifact Classes</span>
            <span class="hero-chip">Cosine OOD Filter</span>
            <span class="hero-chip">CPU Inference</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Awakening the oracle …"):
    md = load_model(CKPT_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙ &nbsp; Configuration</div>',
                unsafe_allow_html=True)

    st.markdown('<div class="sidebar-heading">Top-K Predictions</div>',
                unsafe_allow_html=True)
    top_k = st.slider("Top-K", 1, 5, 3, label_visibility="collapsed")

    st.markdown('<div class="sidebar-heading">OOD Similarity Threshold</div>',
                unsafe_allow_html=True)

    stored_thresh  = md["threshold"]
    default_thresh = round(stored_thresh * 0.50, 4)

    threshold = st.slider(
        "Threshold",
        min_value=0.50,
        max_value=round(min(stored_thresh * 1.05, 0.95), 2),
        value=default_thresh,
        step=0.01,
        label_visibility="collapsed",
    )

    st.markdown(f"""
    <div style="display:flex; justify-content:space-between;
                font-family:'Josefin Sans',sans-serif; font-size:0.55rem;
                letter-spacing:0.12em; color:rgba(191,164,106,0.30);
                margin-top:4px; text-transform:uppercase;">
        <span>Open</span>
        <span style="color:var(--gold); font-weight:400;">{threshold:.3f}</span>
        <span>Strict</span>
    </div>
    <div class="thresh-box">
        Model threshold: {stored_thresh:.4f}<br>
        Effective: {threshold:.4f} &nbsp;({threshold/stored_thresh*100:.0f}% of original)<br>
        Lower → accepts more &nbsp;·&nbsp; Higher → stricter
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:1px; background:var(--border); margin:1.5rem 0;'></div>",
                unsafe_allow_html=True)

    st.markdown('<div class="sidebar-heading">📜 Known Classes</div>',
                unsafe_allow_html=True)

    for i, cn in enumerate(md["class_names"]):
        st.markdown(
            f'<div class="class-item">'
            f'<span class="class-num">{i+1:02d}</span>'
            f'<span class="class-name">{cn}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("""
    <div style="margin-top:2rem; font-family:'Josefin Sans',sans-serif;
                font-size:0.48rem; font-weight:200; letter-spacing:0.18em;
                text-transform:uppercase; color:rgba(139,115,85,0.25);
                text-align:center; line-height:2;">
        Graduation Project<br>Faculty of Artificial Intelligence<br>2025 / 2026
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.55], gap="large")

# ── LEFT — Upload & Image ─────────────────────────────────────────────────
with col_left:
    st.markdown('<div class="sec-label">Upload Artifact Image</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "drop",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed",
    )

    run_btn = False
    if uploaded:
        try:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption=uploaded.name, use_container_width=True)
            run_btn = st.button("𓂀  Classify Artifact", use_container_width=True)
        except Exception as exc:
            st.error(f"Cannot open image: {exc}")
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-glyph">𓁹</div>
            <div class="empty-text">
                Drop a JPG · PNG · WEBP<br>to begin identification
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── RIGHT — Results ───────────────────────────────────────────────────────
with col_right:
    st.markdown('<div class="sec-label">Classification Result</div>',
                unsafe_allow_html=True)

    if uploaded and run_btn:
        with st.spinner("Reading the stone …"):
            t0      = time.time()
            result  = predict(pil_img, md, threshold, top_k=top_k)
            elapsed = time.time() - t0

        sim       = result["similarity"]
        in_domain = result["in_domain"]
        sim_pct   = max(0, min(100, int(sim * 100)))
        fill_cls  = "meter-fill-ok" if in_domain else "meter-fill-bad"
        sim_color = "#2ecc71" if in_domain else "#e74c3c"

        # ── Result card ───────────────────────────────────────────────────
        if in_domain:
            conf     = result["confidence"]
            conf_pct = f"{conf:.1%}"
            st.markdown(f"""
            <div class="card card-known">
                <div class="card-verdict card-verdict-ok">
                    ✦ &nbsp; Artifact Identified &nbsp; ✦
                </div>
                <div class="card-name">{result['predicted']}</div>
                <div class="card-meta">Confidence &nbsp;·&nbsp; {conf_pct}</div>
                <div class="meter-wrap">
                    <div class="meter-row">
                        <span class="meter-label">Domain Similarity</span>
                        <span class="meter-value" style="color:{sim_color};">{sim:.4f}</span>
                    </div>
                    <div class="meter-track">
                        <div class="{fill_cls}" style="width:{sim_pct}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="card card-unknown">
                <div class="card-verdict card-verdict-bad">
                    ✦ &nbsp; Outside Known Domain &nbsp; ✦
                </div>
                <div class="card-name-bad">Artifact Not Recognised</div>
                <div class="card-meta">
                    Similarity {sim:.4f} is below the threshold {threshold:.4f}.<br>
                    Try lowering the threshold slider in the sidebar.
                </div>
                <div class="meter-wrap">
                    <div class="meter-row">
                        <span class="meter-label">Domain Similarity</span>
                        <span class="meter-value" style="color:{sim_color};">{sim:.4f}</span>
                    </div>
                    <div class="meter-track">
                        <div class="{fill_cls}" style="width:{sim_pct}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Metadata chips ────────────────────────────────────────────────
        st.markdown(f"""
        <div class="chip-row">
            <div class="chip">Status &nbsp;<b>{'Known' if in_domain else 'Unknown'}</b></div>
            <div class="chip">Similarity &nbsp;<b>{sim:.4f}</b></div>
            <div class="chip">Threshold &nbsp;<b>{threshold:.4f}</b></div>
            <div class="chip">Inference &nbsp;<b>{elapsed*1000:.0f} ms</b></div>
            <div class="chip">Model &nbsp;<b>EfficientNet-B4</b></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Top-K predictions ─────────────────────────────────────────────
        if in_domain and result.get("topk"):
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sec-label">Top Predictions</div>',
                        unsafe_allow_html=True)

            ranks    = ["I", "II", "III", "IV", "V"]
            max_conf = result["topk"][0]["confidence"]

            for i, item in enumerate(result["topk"]):
                pct  = item["confidence"]
                w    = int(pct / max(max_conf, 1e-6) * 100)
                rank = ranks[i] if i < len(ranks) else "·"
                fill = "bar-fill-gold" if i == 0 else "bar-fill-dim"
                ccls = ("bar-conf-hi" if pct >= 0.40
                        else "bar-conf-mid" if pct >= 0.20 else "bar-conf-lo")
                st.markdown(f"""
                <div class="bar-item">
                    <div class="bar-header">
                        <span>
                            <span class="bar-class-rank">{rank}</span>
                            <span class="bar-class">{item['class']}</span>
                        </span>
                        <span class="bar-conf {ccls}">{pct:.1%}</span>
                    </div>
                    <div class="bar-track">
                        <div class="{fill}" style="width:{w}%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Ensemble note ─────────────────────────────────────────────
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div style="background:rgba(191,164,106,0.04); border:1px solid var(--border);
                        border-radius:8px; padding:1rem 1.2rem;">
                <div class="sec-label" style="margin-bottom:0.6rem;">Ensemble Architecture</div>
                <div style="font-family:'Cormorant Garamond',serif; font-style:italic;
                            font-size:0.85rem; color:var(--sand); line-height:1.7;">
                    EfficientNet-B4 serves as the primary classifier. When confidence is below
                    the priority threshold, three CLIP models (ViT-L/14@336, ViT-B/32, ViT-B/16)
                    vote in zero-shot mode. The final prediction is determined by weighted
                    majority consensus across all four models.
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── RAG status ────────────────────────────────────────────────
            conf = result["confidence"]
            if conf >= 0.30:
                st.markdown("""
                <div class="rag-banner rag-yes">
                    <span class="rag-icon">✅</span>
                    <span class="rag-text rag-text-yes">
                        Send to RAG System &nbsp;·&nbsp; Confidence sufficient for downstream retrieval
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="rag-banner rag-no">
                    <span class="rag-icon">⚠️</span>
                    <span class="rag-text rag-text-no">
                        Low confidence &nbsp;·&nbsp; Verify before sending to RAG system
                    </span>
                </div>
                """, unsafe_allow_html=True)

        # ── Raw output ────────────────────────────────────────────────────
        with st.expander("🔬  Raw JSON Output"):
            st.json(result)

    elif not uploaded:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-glyph">𓂀</div>
            <div class="empty-text">
                Upload an image on the left,<br>then press Classify Artifact
            </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-glyphs">𓀀 𓁀 𓂀 𓃀 𓄀 𓅀 𓆀 𓇀 𓈀 𓉀</div>
    <div class="footer-text">
        Kemet Vision v2.0 &nbsp;·&nbsp;
        EfficientNet-B4 + CLIP Ensemble &nbsp;·&nbsp;
        74 Egyptian Artifact Classes &nbsp;·&nbsp;
        Cosine-Similarity OOD Filter &nbsp;·&nbsp;
        Graduation Project 2025 / 2026
    </div>
</div>
""", unsafe_allow_html=True)