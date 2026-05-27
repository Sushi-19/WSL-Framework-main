import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import os, json, sys
from torchvision import transforms
import torchvision.transforms.functional as TF

# ── Add src to path so model classes can be imported ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from models.baseline import SimpleCNN, ResNet, MLP

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WSL Inference | Image Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark theme CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background-color: #0E1117; }
  [data-testid="stSidebar"]          { background-color: #1C1F26; }
  [data-testid="stHeader"]           { background-color: #0E1117; }
  .main-header {
    font-size: 2.2rem; font-weight: 700; color: #FFFFFF;
    text-align: center; padding: 1rem 0; margin-bottom: 1.5rem;
    border-bottom: 3px solid #00B4D8; font-family: sans-serif;
  }
  .section-header {
    font-size: 1.4rem; font-weight: 600; color: #FFFFFF;
    border-bottom: 2px solid #00B4D8; padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem 0; font-family: sans-serif;
  }
  .result-card {
    background: #1C1F26; border: 2px solid #2A2D35;
    border-radius: 10px; padding: 1.5rem; margin: 0.5rem 0;
    font-family: sans-serif;
  }
  .pred-label {
    font-size: 2rem; font-weight: 800; color: #00B4D8;
    font-family: sans-serif;
  }
  .conf-value {
    font-size: 1.5rem; font-weight: 700; color: #4ECDC4;
    font-family: sans-serif;
  }
  .info-pill {
    display: inline-block; background: #2A2D35; color: #B0B0B0;
    border-radius: 20px; padding: 0.3rem 0.9rem; font-size: 0.85rem;
    margin: 0.2rem; font-family: sans-serif;
  }
  .stButton > button {
    background: linear-gradient(135deg, #00B4D8, #0077B6);
    color: #FFFFFF; border: none; border-radius: 8px;
    padding: 0.75rem 2rem; font-weight: 700; font-size: 1rem;
    transition: all 0.3s ease; font-family: sans-serif; width: 100%;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 180, 216, 0.4);
  }
  .stMarkdown p, .stMarkdown li { color: #FFFFFF; font-family: sans-serif; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
EXPERIMENTS_DIR = os.path.join(os.path.dirname(__file__), "experiments", "matrix_results_100epochs")

DATASET_CONFIG = {
    "CIFAR-100": {
        "key": "cifar100", "num_classes": 100, "img_size": 32,
        "mean": (0.5071, 0.4867, 0.4408), "std": (0.2675, 0.2565, 0.2761),
        "classes": [
            "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
            "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle",
            "chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
            "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard",
            "lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple_tree","motorcycle","mountain",
            "mouse","mushroom","oak_tree","orange","orchid","otter","palm_tree","pear","pickup_truck","pine_tree",
            "plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket",
            "rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider",
            "squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor",
            "train","trout","tulip","turtle","wardrobe","whale","willow_tree","wolf","woman","worm"
        ]
    },
    "STL-10": {
        "key": "stl10", "num_classes": 10, "img_size": 32,
        "mean": (0.4467, 0.4398, 0.4066), "std": (0.2603, 0.2566, 0.2713),
        "classes": ["airplane","bird","car","cat","deer","dog","horse","monkey","ship","truck"]
    },
    "Animal-10N": {
        "key": "animal10n", "num_classes": 10, "img_size": 32,
        "mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010),
        "classes": ["cat","lynx","wolf","coyote","cheetah","jaguar","chimpanzee","orangutan","hamster","guinea_pig"]
    },
    "CIFAR-10N": {
        "key": "cifar10n", "num_classes": 10, "img_size": 32,
        "mean": (0.4914, 0.4822, 0.4465), "std": (0.2023, 0.1994, 0.2010),
        "classes": ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    },
    "SVHN": {
        "key": "svhn", "num_classes": 10, "img_size": 32,
        "mean": (0.4377, 0.4438, 0.4728), "std": (0.1980, 0.2010, 0.1970),
        "classes": ["0","1","2","3","4","5","6","7","8","9"]
    }
}

MODEL_KEYS = {
    "Simple CNN": "simple_cnn",
    "ResNet":     "resnet",
    "MLP":        "mlp"
}

STRATEGY_KEYS = {
    "Baseline":                    "baseline",
    "Consistency Regularization":  "consistency",
    "Pseudo-Labeling":             "pseudo_labeling",
    "Co-Training":                 "co_training",
    "ADAS-WSL":                    "adas_wsl",
    "Combined (Fixed)":            "combined"
}

STRATEGY_DESC = {
    "Baseline":                    "Standard supervised learning with clean labels.",
    "Consistency Regularization":  "Teacher-student learning; penalises inconsistent predictions under augmentation.",
    "Pseudo-Labeling":             "Assigns pseudo-labels to unlabelled samples using model confidence.",
    "Co-Training":                 "Multi-view ensemble; two models supervise each other.",
    "ADAS-WSL":                    "Adaptive Dual-Axis WSL — noise-aware, confidence-driven curriculum.",
    "Combined (Fixed)":            "Uses fixed lambda weights [0.4, 0.3, 0.3] for pseudo-labeling, consistency, and co-training without dynamic PASW."
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Helpers ───────────────────────────────────────────────────────────────────
# Only checkpoints that actually exist on disk
AVAILABLE_CHECKPOINTS = {}
for ds_name, ds_cfg in DATASET_CONFIG.items():
    for mod_name, mod_key in MODEL_KEYS.items():
        for strat_name, strat_key in STRATEGY_KEYS.items():
            folder = f"{ds_cfg['key']}_{mod_key}_{strat_key}"
            ckpt_path = os.path.join(EXPERIMENTS_DIR, folder, "best_model.pt")
            if os.path.exists(ckpt_path):
                AVAILABLE_CHECKPOINTS[(ds_name, mod_name, strat_name)] = ckpt_path

def get_model_path(dataset: str, model: str, strategy: str):
    """Return (path, is_available) for the requested combination."""
    key = (dataset, model, strategy)
    if key in AVAILABLE_CHECKPOINTS:
        return AVAILABLE_CHECKPOINTS[key], True
    return None, False


@st.cache_resource(show_spinner=False)
def load_model(dataset: str, model_name: str, strategy: str):
    cfg       = DATASET_CONFIG[dataset]
    n_classes = cfg["num_classes"]
    img_size  = cfg["img_size"]

    path, available = get_model_path(dataset, model_name, strategy)
    if not available:
        return None, "NO_CHECKPOINT"

    if model_name == "Simple CNN":
        model = SimpleCNN(num_classes=n_classes)
    elif model_name == "ResNet":
        model = ResNet(num_classes=n_classes, in_channels=3)
    else:
        flat  = 3 * img_size * img_size
        model = MLP(input_size=flat, num_classes=n_classes)

    try:
        ckpt  = torch.load(path, map_location=DEVICE, weights_only=False)
        state = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        # Strip 'model1.' prefix if saved from a UnifiedWSLModel wrapper
        if any(k.startswith("model1.") for k in state.keys()):
            state = {k.replace("model1.", "", 1): v
                     for k, v in state.items() if k.startswith("model1.")}
        model.load_state_dict(state, strict=False)
        model.to(DEVICE).eval()
        return model, None
    except Exception as e:
        return None, str(e)


def preprocess(image: Image.Image, dataset: str, model_name: str):
    """Preprocessing fixed to match training pipeline (0.5 mean/std and exact size resize)."""
    cfg  = DATASET_CONFIG[dataset]
    size = cfg["img_size"]
    mean = (0.5, 0.5, 0.5)
    std  = (0.5, 0.5, 0.5)

    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform(image.convert("RGB")).unsqueeze(0).float()


@torch.no_grad()
def run_inference(model, tensor, temperature: float = 1.5):
    """Improvement #3: Temperature scaling for better-calibrated confidence."""
    tensor = tensor.to(DEVICE)
    logits = model(tensor)
    probs  = F.softmax(logits / temperature, dim=1).squeeze().cpu().numpy()
    return probs


@torch.no_grad()
def run_inference_tta(model, image: Image.Image, dataset: str, temperature: float = 1.5):
    """Improvement #1: Test-Time Augmentation — average over 6 augmented views."""
    cfg  = DATASET_CONFIG[dataset]
    size = cfg["img_size"]
    mean = (0.5, 0.5, 0.5)
    std  = (0.5, 0.5, 0.5)

    norm = transforms.Normalize(mean=mean, std=std)
    to_t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        norm,
    ])

    def augment(pil_img):
        return to_t(pil_img.convert("RGB")).unsqueeze(0).float().to(DEVICE)

    rgb = image.convert("RGB")
    augmented_views = [
        augment(rgb),                              # original
        augment(TF.hflip(rgb)),                    # horizontal flip
        augment(TF.adjust_brightness(rgb, 1.15)), # brighter
        augment(TF.adjust_contrast(rgb, 1.15)),   # more contrast
        augment(TF.rotate(rgb, 10)),               # rotate +10°
        augment(TF.rotate(rgb, -10)),              # rotate -10°
    ]

    all_probs = [
        F.softmax(model(v) / temperature, dim=1).squeeze().cpu().numpy()
        for v in augmented_views
    ]
    return np.mean(all_probs, axis=0)  # averaged probability distribution


def run_ensemble_inference(image: Image.Image, dataset: str, model_name: str,
                           primary_strategy: str, use_tta: bool = True):
    """Improvement #2: Ensemble across all available strategies for the chosen model."""
    ensemble_probs = []
    strategies_used = []

    tensor = preprocess(image, dataset, model_name)

    for strat_name in STRATEGY_KEYS:
        m, err = load_model(dataset, model_name, strat_name)
        if m is not None and err is None:
            if use_tta:
                p = run_inference_tta(m, image, dataset)
            else:
                p = run_inference(m, tensor)
            ensemble_probs.append(p)
            strategies_used.append(strat_name)

    if not ensemble_probs:
        return None, []
    return np.mean(ensemble_probs, axis=0), strategies_used


def bar_chart(probs, classes, top_k=10):
    idx   = np.argsort(probs)[::-1][:top_k]
    vals  = probs[idx]
    names = [classes[i] for i in idx]

    colors = ["#00B4D8" if i == 0 else "#2A2D35" for i in range(len(vals))]

    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in vals],
        textposition="outside",
        textfont=dict(color="#FFFFFF", size=12)
    ))
    fig.update_layout(
        height=max(300, top_k * 36),
        margin=dict(l=10, r=60, t=10, b=10),
        plot_bgcolor="#1C1F26", paper_bgcolor="#0E1117",
        xaxis=dict(range=[0, min(1.05, vals[0] + 0.15)],
                   gridcolor="#2A2D35", zerolinecolor="#2A2D35",
                   tickformat=".0%", tickfont=dict(color="#FFFFFF")),
        yaxis=dict(autorange="reversed", tickfont=dict(color="#FFFFFF")),
        font=dict(family="sans-serif", color="#FFFFFF")
    )
    return fig

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="main-header">🔍 WSL Image Inference</h1>', unsafe_allow_html=True)

# Sidebar controls
with st.sidebar:
    st.markdown("### ⚙️ Model Configuration")

    dataset  = st.selectbox("Dataset", list(DATASET_CONFIG.keys()),
                            help="Dataset the model was trained on")
    model_nm = st.selectbox("Model Architecture", list(MODEL_KEYS.keys()))
    strategy = st.selectbox("WSL Strategy", list(STRATEGY_KEYS.keys()))

    st.markdown("---")
    st.markdown(f"""
    <div style="background:#1C1F26;border-radius:8px;padding:1rem;border:1px solid #2A2D35;">
      <div style="color:#B0B0B0;font-size:0.82rem;font-family:sans-serif;">
        <strong style="color:#00B4D8;">Strategy info</strong><br><br>
        {STRATEGY_DESC[strategy]}
      </div>
    </div>
    """, unsafe_allow_html=True)

    _, available = get_model_path(dataset, model_nm, strategy)
    status_color = "#4ECDC4" if available else "#FF6B6B"
    status_text  = "✅ Checkpoint available" if available else "❌ No saved checkpoint"
    st.markdown(f"""
    <div style="margin-top:1rem;padding:0.75rem;background:#1C1F26;border-radius:8px;
         border-left:4px solid {status_color};font-family:sans-serif;font-size:0.85rem;color:{status_color};">
      {status_text}
    </div>
    """, unsafe_allow_html=True)

    if not available:
        st.markdown("""
        <div style="margin-top:0.5rem;padding:0.6rem;background:#1C1F26;border-radius:8px;
             font-family:sans-serif;font-size:0.78rem;color:#B0B0B0;
             border:1px solid #2A2D35;">
          ⚡ The model checkpoint (<code style="color:#00B4D8;">best_model.pt</code>) for this combination was not found in the experiments folder.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Device**")
    st.code(str(DEVICE), language=None)

# Main area
col_upload, col_result = st.columns([1, 1], gap="large")

with col_upload:
    st.markdown('<div class="section-header">📤 Upload Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.markdown(f"""
        <div style="margin-top:0.5rem;">
          <span class="info-pill">📐 {image.width} × {image.height} px</span>
          <span class="info-pill">🎨 {image.mode}</span>
          <span class="info-pill">📦 {uploaded.size // 1024} KB</span>
        </div>
        """, unsafe_allow_html=True)

        # Resize preview
        cfg = DATASET_CONFIG[dataset]
        preview = image.convert("RGB").resize((cfg["img_size"], cfg["img_size"]), Image.LANCZOS)
        st.caption(f"↳ Will be resized to {cfg['img_size']}×{cfg['img_size']} for inference")

        # Options
        st.markdown("---")
        use_tta      = st.toggle("🔁 Test-Time Augmentation (TTA)", value=True,
                                 help="Average predictions over 6 augmented views for higher accuracy")
        use_ensemble = st.toggle("🧩 Ensemble across all strategies", value=False,
                                 help="Average predictions from all available strategy checkpoints")
        temperature  = st.slider("🌡️ Temperature (confidence calibration)", 0.5, 3.0, 1.5, 0.1,
                                 help="Higher = more uncertain/honest confidence. Lower = sharper but overconfident.")

        run_btn = st.button("🚀 Run Inference", type="primary")
    else:
        st.info("⬆️ Please upload an image to begin.")
        run_btn = False

with col_result:
    st.markdown('<div class="section-header">📊 Prediction Results</div>', unsafe_allow_html=True)

    if uploaded and run_btn:
        classes = cfg["classes"]
        probs   = None
        strategies_used = [strategy]

        if use_ensemble:
            # Improvement #2: Ensemble across all available strategy checkpoints
            with st.spinner(f"Running ensemble inference across all strategies ({model_nm} on {dataset})…"):
                probs, strategies_used = run_ensemble_inference(
                    image, dataset, model_nm, strategy, use_tta=use_tta
                )
            if probs is None:
                st.error("❌ No checkpoints found for any strategy. Cannot run ensemble.")
        else:
            with st.spinner(f"Loading {model_nm} ({strategy}) checkpoint…"):
                model, err = load_model(dataset, model_nm, strategy)

            if err == "NO_CHECKPOINT":
                from wsl_streamlit_app import PERFORMANCE_DATA, STRATEGY_KEY_MAP
                st_key  = STRATEGY_KEY_MAP.get(strategy, "baseline")
                known   = PERFORMANCE_DATA.get(dataset, {}).get(model_nm, {}).get(st_key)
                acc_str = f"{known*100:.2f}%" if known else "N/A"
                st.warning(
                    f"**No saved checkpoint** for **{dataset} · {model_nm} · {strategy}**.\n\n"
                    f"The `best_model.pt` file is missing from your experiments folder.\n\n"
                    f"📊 Known test accuracy from experiment logs: **{acc_str}**"
                )
            elif err:
                st.error(f"**Failed to load model:**\n\n{err}")
            else:
                with st.spinner("Running inference…"):
                    if use_tta:
                        # Improvement #1: TTA
                        probs = run_inference_tta(model, image, dataset, temperature=temperature)
                    else:
                        tensor = preprocess(image, dataset, model_nm)
                        probs  = run_inference(model, tensor, temperature=temperature)

        if probs is not None:
            top_idx    = int(np.argmax(probs))
            top_label  = classes[top_idx]
            top_conf   = probs[top_idx]
            top2_idx   = int(np.argsort(probs)[::-1][1])
            top2_label = classes[top2_idx]
            top2_conf  = probs[top2_idx]

            # Improvement #5: Low-confidence warning
            if top_conf < 0.50:
                st.warning(
                    f"⚠️ **Low confidence ({top_conf*100:.1f}%).** The model is uncertain. "
                    f"Try a clearer image, enable TTA, or switch to a different model/strategy."
                )

            # Ensemble badge
            mode_badge = ""
            if use_ensemble:
                mode_badge = f"<span class='info-pill'>🧩 Ensemble · {len(strategies_used)} strategies</span>"
            elif use_tta:
                mode_badge = "<span class='info-pill'>🔁 TTA · 6 augmentations</span>"

            # Result card
            st.markdown(f"""
            <div class="result-card">
              <div style="color:#B0B0B0;font-size:0.85rem;margin-bottom:0.5rem;">Top Prediction</div>
              <div class="pred-label">{top_label.replace("_"," ").title()}</div>
              <div class="conf-value">{top_conf*100:.2f}% confidence</div>
              <div style="margin-top:0.4rem;">{mode_badge}</div>
              <hr style="border-color:#2A2D35;margin:1rem 0;">
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;color:#B0B0B0;font-size:0.85rem;">
                <div><strong>Runner-up:</strong> {top2_label.replace("_"," ").title()}</div>
                <div><strong>Runner-up conf:</strong> {top2_conf*100:.2f}%</div>
                <div><strong>Dataset:</strong> {dataset}</div>
                <div><strong>Model:</strong> {model_nm}</div>
                <div style="grid-column:1/-1"><strong>Strategy:</strong> {strategy}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Top-K bar chart
            st.markdown('<div class="section-header">📈 Top-10 Class Probabilities</div>',
                        unsafe_allow_html=True)
            top_k = min(10, len(classes))
            st.plotly_chart(bar_chart(probs, classes, top_k), use_container_width=True)

            # Ensemble breakdown (if applicable)
            if use_ensemble and len(strategies_used) > 1:
                with st.expander(f"🧩 Ensemble strategies used ({len(strategies_used)})"):
                    for s in strategies_used:
                        st.markdown(f"- **{s}**: {STRATEGY_DESC[s]}")

            # Full probability table (expander)
            with st.expander("📋 All class probabilities"):
                import pandas as pd
                df = pd.DataFrame({
                    "Class": [c.replace("_", " ").title() for c in classes],
                    "Probability": probs
                }).sort_values("Probability", ascending=False).reset_index(drop=True)
                df["Probability"] = df["Probability"].map(lambda x: f"{x*100:.3f}%")
                st.dataframe(df, use_container_width=True, height=300)

    elif not uploaded:
        st.markdown("""
        <div style="height:300px;display:flex;align-items:center;justify-content:center;
             background:#1C1F26;border-radius:10px;border:2px dashed #2A2D35;">
          <div style="text-align:center;color:#B0B0B0;font-family:sans-serif;">
            <div style="font-size:3rem;">🖼️</div>
            <div style="margin-top:0.5rem;">Upload an image to see predictions</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#B0B0B0;padding:1.5rem 0;font-family:sans-serif;'>
  <strong style="color:#FFFFFF;">WSL Inference App</strong> &nbsp;·&nbsp;
  Weakly Supervised Learning Framework &nbsp;·&nbsp; Developed by Mustqeem Sannakki
</div>
""", unsafe_allow_html=True)
