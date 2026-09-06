import os
import re
import base64
import unicodedata
from PIL import Image
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(
    page_title="Apex Motors | دور على عربيتك",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# صورة العربية المستخدمة كخلفية كاملة للواجهة
@st.cache_data(show_spinner=False)
def get_background_image():
    image_path = "hero_car.jpg"
    if not os.path.exists(image_path):
        return ""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"

BG_IMAGE = get_background_image()

# ==============================================================================
# CSS: شريط بحث مدمج في المنتصف مع أيقونة الكاميرا بالداخل
# ==============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    :root {{
        --accent: #f4f4f5;
        --red: #ff3b3b;
        --red-soft: rgba(255,59,59,0.22);
        --glass: rgba(5,7,10,0.72);
        --line: rgba(255,255,255,0.15);
    }}

    html, body, [data-testid="stAppViewContainer"] {{
        background: #050608 !important;
    }}

    .stApp {{
        background: #050608 !important;
        color: #f8fafc;
        font-family: 'Cairo', 'Plus Jakarta Sans', sans-serif;
    }}

    /* =========================================================
       CAR IMAGE — الصورة نفسها واضحة في الخلفية
       ========================================================= */
    .stApp::before {{
        content: "";
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background:
            linear-gradient(90deg,
                rgba(3,4,6,0.70) 0%,
                rgba(3,4,6,0.34) 42%,
                rgba(3,4,6,0.12) 100%);
    }}

    .car-background {{
        position: fixed;
        inset: 0;
        z-index: 0;
        pointer-events: none;
        background-image: url("{BG_IMAGE}");
        background-repeat: no-repeat;
        background-position: 78% 42%;
        background-size: auto 88vh;
        filter: brightness(1.32) contrast(1.12);
        opacity: 0.92;
    }}

    .car-background::after {{
        content: "";
        position: absolute;
        inset: 0;
        background:
            linear-gradient(180deg,
                rgba(4,5,7,0.08) 0%,
                rgba(4,5,7,0.06) 48%,
                rgba(4,5,7,0.62) 100%);
    }}

    .main .block-container {{
        position: relative;
        z-index: 2;
        padding-top: 0.4rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }}

    /* =========================================================
       HERO — شكل مقدمة عربية رياضية بدل المربع
       ========================================================= */
    .hero-box {{
        position: relative;
        width: min(900px, 94%);
        min-height: 335px;
        margin: 25px auto 32px;
        padding: 64px 60px 55px;
        text-align: center;
        overflow: hidden;

        /* silhouette لواجهة سيارة */
        clip-path: polygon(
            7% 77%,
            10% 47%,
            18% 38%,
            27% 17%,
            36% 7%,
            64% 7%,
            73% 17%,
            82% 38%,
            90% 47%,
            93% 77%,
            86% 91%,
            14% 91%
        );

        background:
            linear-gradient(180deg,
                rgba(31,33,37,0.93) 0%,
                rgba(9,10,13,0.96) 62%,
                rgba(3,4,6,0.98) 100%);
        box-shadow:
            0 28px 80px rgba(0,0,0,0.62),
            inset 0 1px 0 rgba(255,255,255,0.13);
    }}

    /* لمعة على سقف العربية */
    .hero-box::before {{
        content: "";
        position: absolute;
        left: 27%;
        right: 27%;
        top: 20px;
        height: 72px;
        border-radius: 50%;
        background: linear-gradient(180deg,
            rgba(255,255,255,0.16),
            rgba(255,255,255,0));
        filter: blur(3px);
        pointer-events: none;
    }}

    /* خط الشبكة الأمامية */
    .hero-box::after {{
        content: "";
        position: absolute;
        left: 37%;
        right: 37%;
        bottom: 30px;
        height: 10px;
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 0 0 18px 18px;
        box-shadow:
            0 4px 0 rgba(255,255,255,0.05),
            0 0 25px rgba(255,59,59,0.18);
        pointer-events: none;
    }}

    .hero-lights {{
        position: absolute;
        top: 120px;
        left: 13%;
        right: 13%;
        display: flex;
        justify-content: space-between;
        pointer-events: none;
    }}

    .hero-light {{
        width: 86px;
        height: 22px;
        border-radius: 50%;
        background: linear-gradient(90deg, transparent, #fff, transparent);
        box-shadow:
            0 0 8px #fff,
            0 0 25px rgba(255,255,255,0.7),
            0 0 45px rgba(255,59,59,0.18);
        opacity: 0.9;
    }}

    .hero-content {{
        position: relative;
        z-index: 5;
    }}

    .hero-kicker {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 7px 16px;
        margin-bottom: 13px;
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 999px;
        background: rgba(0,0,0,0.48);
        color: #d4d4d8;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.68rem;
        font-weight: 800;
        letter-spacing: 2px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.35);
    }}

    .hero-title {{
        margin: 0;
        color: #ffffff;
        font-family: 'Plus Jakarta Sans', 'Cairo', sans-serif;
        font-size: clamp(2.6rem, 6vw, 5rem);
        line-height: 1;
        font-weight: 800;
        letter-spacing: -3px;
        text-shadow: 0 7px 25px rgba(0,0,0,0.75);
    }}

    .hero-title span {{
        color: var(--red);
        text-shadow: 0 0 28px rgba(255,59,59,0.2);
    }}

    .hero-subtitle {{
        color: #e4e4e7;
        font-size: 0.98rem;
        max-width: 650px;
        margin: 15px auto 0;
        line-height: 1.85;
        text-shadow: 0 3px 15px rgba(0,0,0,0.95);
    }}

    /* =========================================================
       SEARCH — جزء من شكل العربية
       ========================================================= */
    [data-testid="stTextInput"] {{
        position: relative;
        z-index: 10;
        width: min(720px, 100%);
        margin: 0 auto !important;
        padding: 9px !important;
        border: 1px solid rgba(255,255,255,0.18) !important;
        border-radius: 20px !important;
        background: rgba(7,8,10,0.84) !important;
        box-shadow:
            0 22px 55px rgba(0,0,0,0.58),
            inset 0 1px 0 rgba(255,255,255,0.08) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
    }}

    [data-testid="stTextInput"]::before {{
        content: "";
        position: absolute;
        left: 50%;
        top: -2px;
        width: 120px;
        height: 3px;
        transform: translateX(-50%);
        background: linear-gradient(90deg, transparent, var(--red), transparent);
        border-radius: 999px;
        box-shadow: 0 0 18px rgba(255,59,59,0.5);
    }}

    [data-testid="stTextInput"] input {{
        background: rgba(0,0,0,0.56) !important;
        border: 1px solid rgba(255,255,255,0.16) !important;
        color: #ffffff !important;
        border-radius: 14px !important;
        height: 58px !important;
        padding-left: 22px !important;
        padding-right: 68px !important;
        font-size: 1.02rem !important;
        box-shadow: inset 0 2px 12px rgba(0,0,0,0.48) !important;
        transition: 0.25s ease !important;
        direction: rtl;
        text-align: right;
    }}

    [data-testid="stTextInput"] input::placeholder {{
        color: #b8bac0 !important;
    }}

    [data-testid="stTextInput"] input:focus {{
        border-color: rgba(255,255,255,0.38) !important;
        box-shadow:
            0 0 0 3px rgba(255,255,255,0.045),
            0 0 24px rgba(255,59,59,0.12),
            inset 0 2px 12px rgba(0,0,0,0.48) !important;
    }}

    /* رفع الصورة كزر كاميرا داخل البحث */
    [data-testid="stFileUploader"] {{
        margin-top: -58px !important;
        height: 58px !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        padding-right: 18px !important;
        pointer-events: none !important;
        border: none !important;
        background: transparent !important;
        position: relative;
        z-index: 20;
    }}

    [data-testid="stFileUploader"] section {{
        padding: 0 !important;
        min-height: unset !important;
        border: none !important;
        background: transparent !important;
        pointer-events: auto !important;
    }}

    [data-testid="stFileUploaderDropzone"] {{
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
    }}

    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzone"] > div:not(:has(button)) {{
        display: none !important;
    }}

    [data-testid="stFileUploader"] button {{
        width: 40px !important;
        height: 40px !important;
        border-radius: 12px !important;
        background: rgba(255,255,255,0.07) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        box-shadow: 0 4px 14px rgba(0,0,0,0.3) !important;
        padding: 4px !important;
        cursor: pointer !important;
        color: #fff !important;
        transition: 0.2s ease !important;
    }}

    [data-testid="stFileUploader"] button:hover {{
        transform: translateY(-2px) scale(1.06);
        background: rgba(255,59,59,0.16) !important;
        border-color: rgba(255,59,59,0.7) !important;
        box-shadow: 0 0 22px rgba(255,59,59,0.2) !important;
    }}

    [data-testid="stFileUploader"] button::before {{
        content: "📷";
        font-size: 1.1rem;
    }}

    [data-testid="stFileUploader"] button span,
    [data-testid="stFileUploader"] button p {{
        display: none !important;
    }}

    [data-testid="stFileUploaderFile"] {{
        margin-top: 14px !important;
        background: rgba(7,8,10,0.88) !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 12px !important;
    }}

    /* =========================================================
       RESULTS — نفس الـconcept بدون تغيير في الـlogic
       ========================================================= */
    .car-card {{
        background: rgba(6,8,11,0.82);
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 16px;
        transition: 0.2s ease;
        box-shadow: 0 14px 35px rgba(0,0,0,0.5);
        backdrop-filter: blur(13px);
        -webkit-backdrop-filter: blur(13px);
    }}

    .car-card:hover {{
        border-color: rgba(255,255,255,0.28);
        transform: translateY(-2px);
    }}

    .deal-badge-great {{
        background: rgba(34,197,94,0.13);
        border: 1px solid rgba(34,197,94,0.65);
        color: #86efac;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
    }}

    .deal-badge-overpriced {{
        background: rgba(239,68,68,0.13);
        border: 1px solid rgba(239,68,68,0.65);
        color: #fca5a5;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
    }}

    .deal-badge-fair {{
        background: rgba(234,179,8,0.13);
        border: 1px solid rgba(234,179,8,0.65);
        color: #fde047;
        font-weight: 700;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
    }}

    @media (max-width: 700px) {{
        .car-background {{
            background-size: auto 68vh;
            background-position: 72% 26%;
            opacity: 0.82;
        }}

        .hero-box {{
            min-height: 300px;
            padding: 58px 28px 48px;
            clip-path: polygon(
                5% 79%, 9% 49%, 17% 39%, 28% 16%,
                37% 7%, 63% 7%, 72% 16%, 83% 39%,
                91% 49%, 95% 79%, 86% 92%, 14% 92%
            );
        }}

        .hero-light {{
            width: 55px;
            height: 17px;
        }}

        .hero-title {{
            font-size: 2.65rem;
            letter-spacing: -1.8px;
        }}

        .hero-subtitle {{
            font-size: 0.88rem;
        }}
    }}
</style>
""", unsafe_allow_html=True)

# صورة الخلفية فوق طبقة التطبيق وتحت كل عناصر الواجهة
st.markdown('<div class="car-background"></div>', unsafe_allow_html=True)

# ==============================================================================
# 1. تحميل النماذج والبيانات
# ==============================================================================
@st.cache_resource(show_spinner=False)
def load_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df_data = pd.read_csv("hatla2ee_cleaned_data.csv")
    for col in ["brand", "model", "location", "search_text"]:
        if col in df_data.columns:
            df_data[col] = df_data[col].fillna("").astype(str)

    pricing_model = joblib.load("car_price_model.pkl")
    tfidf_vectorizer = joblib.load("car_tfidf.pkl")
    matrix = tfidf_vectorizer.transform(df_data["search_text"])

    nlp_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = nlp_model.encode(
        df_data["search_text"].tolist(),
        batch_size=64,
        show_progress_bar=False,
        normalize_embeddings=True
    )

    vision_name = "dima806/car_models_image_detection"
    processor = AutoImageProcessor.from_pretrained(vision_name)
    v_model = AutoModelForImageClassification.from_pretrained(vision_name).to(device)
    v_model.eval()

    return device, df_data, pricing_model, tfidf_vectorizer, matrix, nlp_model, embeddings, processor, v_model

try:
    (device, df_recommend, full_pricing_pipeline, tfidf, tfidf_matrix,
     st_model, car_text_embeddings, img_processor, car_vision_model) = load_all()
except Exception as e:
    st.error(f"⚠️ خطأ في تحميل ملفات المشروع: {e}")
    st.stop()

# ==============================================================================
# 2. دوال المعالجة والـ NLP
# ==============================================================================
CITY_MAP = {
    "القاهرة": "cairo", "cairo": "cairo", "اسكندرية": "alexandria", "إسكندرية": "alexandria",
    "التجمع": "tagamo3", "تجمع": "tagamo3", "مدينة نصر": "nasr city", "الجيزة": "giza",
    "زايد": "zayed", "اكتوبر": "october", "كفر الدوار": "kafr el-dawwar", "المنصورة": "mansoura"
}

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text.lower())
    return text.replace("ونص", ".5").replace("وربع", ".25")

def extract_budget(text):
    def scale(v, unit):
        if unit in ("m", "مليون"): return v * 1_000_000
        if unit in ("k", "الف", "ألف"): return v * 1_000
        return v
    m = re.search(r'(?:من\s*)?(\d+(?:\.\d+)?)\s*(m|مليون|k|الف|ألف)?\s*(?:-|to|حتى|لحد|الى|إلى|لـ)\s*(\d+(?:\.\d+)?)\s*(m|مليون|k|الف|ألف)?', text)
    if m:
        return scale(float(m.group(1)), m.group(2)), scale(float(m.group(3)), m.group(4))
    m = re.search(r'(?:under|below|less than|حتى|لحد|اقل من|أقل من|تحت)\s*(\d+(?:\.\d+)?)\s*(m|مليون|k|الف|ألف)?', text)
    if m:
        return None, scale(float(m.group(1)), m.group(2))
    return None, None

def parse_query_to_filters(query, catalog_df):
    text = normalize_text(query)
    filters = {}
    min_p, max_p = extract_budget(text)
    if min_p is not None: filters["min_price"] = min_p
    if max_p is not None: filters["max_price"] = max_p

    if "brand" in catalog_df.columns:
        brands = [b for b in catalog_df["brand"].dropna().unique() if str(b).strip()]
        for b in sorted(brands, key=len, reverse=True):
            if str(b).lower() in text:
                filters["brand"] = b
                break

    if "location" in catalog_df.columns:
        for ar_key, mapped_val in CITY_MAP.items():
            if ar_key in text:
                for loc in catalog_df["location"].dropna().unique():
                    if mapped_val in str(loc).lower():
                        filters["location"] = loc
                        break
                break
    return filters

def predict_vision_top5(image_pil, top_k=5):
    inputs = img_processor(images=image_pil.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = car_vision_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    k = min(top_k, probs.shape[-1])
    top_probs, top_indices = torch.topk(probs, k)
    return [{"label": car_vision_model.config.id2label[idx.item()], "confidence": float(p.item())}
            for p, idx in zip(top_probs, top_indices)]

# ==============================================================================
# 3. واجهة البحث في المنتصف تماماً
# ==============================================================================
st.markdown("""
<div class="hero-box">
    <div class="hero-lights">
        <span class="hero-light"></span>
        <span class="hero-light"></span>
    </div>
    <div class="hero-content">
        <div class="hero-kicker">✦ SMART CAR MARKET</div>
        <div class="hero-title">Apex <span>Motors</span></div>
        <div class="hero-subtitle">اكتشف عربيتك المناسبة بذكاء — ابحث بالمواصفات أو ارفع صورة للسيارة</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ضبط الشريط في منتصف الشاشة بدقة
_, col_search, _ = st.columns([1, 2.6, 1])

with col_search:
    user_query = st.text_input(
        "Search",
        placeholder="اكتب مواصفات العربية واضغط Enter...",
        label_visibility="collapsed"
    )
    uploaded_file = st.file_uploader(
        "Upload",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

# ==============================================================================
# 4. استخراج النتائج عند الضغط على Enter أو رفع صورة
# ==============================================================================
has_query = bool(user_query.strip())
has_image = uploaded_file is not None

if has_query or has_image:
    sub_df = df_recommend.copy()
    filters = {}

    if has_query:
        filters = parse_query_to_filters(user_query, df_recommend)
        if "price" in sub_df.columns:
            if "min_price" in filters: sub_df = sub_df[sub_df["price"] >= filters["min_price"]]
            if "max_price" in filters: sub_df = sub_df[sub_df["price"] <= filters["max_price"]]
        for col in ["brand", "location"]:
            if col in filters and col in sub_df.columns:
                sub_df = sub_df[sub_df[col].astype(str).str.lower() == str(filters[col]).lower()]

    if has_image:
        try:
            pil_img = Image.open(uploaded_file)
            preds = predict_vision_top5(pil_img)
            
            top_car_label = preds[0]['label']
            st.markdown(f"""
            <div style="text-align: center; margin-top: 15px; margin-bottom: 20px;">
                <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #bae6fd; padding: 6px 18px; border-radius: 20px; font-size: 0.95rem;">
                    📷 تم التعرف على السيارة: <strong>{top_car_label}</strong> ({preds[0]['confidence']*100:.1f}%)
                </span>
            </div>
            """, unsafe_allow_html=True)

            if "brand" in sub_df.columns and "model" in sub_df.columns:
                matched_rows = pd.DataFrame()
                for p in preds:
                    lbl = p["label"].lower()
                    mask = (
                        sub_df["brand"].fillna("").astype(str).str.lower().apply(lambda b: bool(b) and b in lbl)
                        | sub_df["model"].fillna("").astype(str).str.lower().apply(lambda m: bool(m) and m in lbl)
                    )
                    subset = sub_df[mask].copy()
                    if not subset.empty:
                        subset["vision_confidence"] = p["confidence"]
                        matched_rows = pd.concat([matched_rows, subset])
                
                dedup_cols = [c for c in ["brand", "model", "year"] if c in matched_rows.columns]
                if not matched_rows.empty and dedup_cols:
                    sub_df = matched_rows.drop_duplicates(subset=dedup_cols).copy()
                elif not matched_rows.empty:
                    sub_df = matched_rows.copy()
        except Exception:
            pass

    if sub_df.empty:
        sub_df = df_recommend.copy()

    if has_query:
        sub_pos = [df_recommend.index.get_loc(i) for i in sub_df.index]
        q_emb = st_model.encode([user_query], normalize_embeddings=True)[0]
        sem_scores = np.dot(car_text_embeddings[sub_pos], q_emb)
        q_tfidf = tfidf.transform([user_query])
        tfidf_scores = cosine_similarity(q_tfidf, tfidf_matrix[sub_pos])[0]
        nlp_score = (0.75 * sem_scores + 0.25 * tfidf_scores).clip(0, 1) * 100
    else:
        nlp_score = 100.0

    sub_df = sub_df.copy()
    if "vision_confidence" in sub_df.columns:
        sub_df["match_score"] = (0.60 * nlp_score + 0.40 * (sub_df["vision_confidence"] * 100)).round(2)
    else:
        sub_df["match_score"] = np.round(nlp_score, 2)

    top_results = sub_df.sort_values("match_score", ascending=False).head(6).copy()

    # التثمين الآمن
    try:
        pred_features = list(full_pricing_pipeline.feature_names_in_)
    except AttributeError:
        pred_features = list(full_pricing_pipeline.named_steps["preprocessor"].feature_names_in_)

    eval_df = top_results[[f for f in pred_features if f in top_results.columns]].copy()
    for col in eval_df.columns:
        if col in ["year", "mileage", "engine_capacity", "horsepower", "car_age", "km_per_year"]:
            eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")

    pred_log = full_pricing_pipeline.predict(eval_df)
    top_results["predicted_fair_price"] = np.expm1(pred_log).round(0)
    top_results["price_difference"] = (top_results["price"] - top_results["predicted_fair_price"]).round(0)

    safe_fair_price = top_results["predicted_fair_price"].replace(0, np.nan)
    pct_diff = (top_results["price_difference"] / safe_fair_price).fillna(0)
    top_results["deal_label"] = np.select(
        [pct_diff <= -0.08, pct_diff >= 0.08],
        ["Great Deal", "Overpriced"],
        default="Fair Price"
    )

    _, col_results, _ = st.columns([1, 2.6, 1])
    with col_results:
        st.write("### أفضل النتائج المطابقة:")

        for _, row in top_results.iterrows():
            b_name = row.get("brand", "")
            m_name = row.get("model", "")
            y_val = int(row.get("year", 0)) if pd.notna(row.get("year")) and row.get("year") != 0 else ""
            price_val = int(row.get("price", 0)) if pd.notna(row.get("price")) else 0
            fair_val = int(row.get("predicted_fair_price", 0)) if pd.notna(row.get("predicted_fair_price")) else 0
            deal_tag = row.get("deal_label", "Fair Price")
            loc_val = row.get("location", "مصر")
            item_link = row.get("item_url", "#")

            if deal_tag == "Great Deal":
                badge_html = '<span class="deal-badge-great">🟢 لقطة (سعر ممتاز)</span>'
            elif deal_tag == "Overpriced":
                badge_html = '<span class="deal-badge-overpriced">🔴 سعر مبالغ فيه</span>'
            else:
                badge_html = '<span class="deal-badge-fair">🟡 سعر عادل</span>'

            st.markdown(f"""
            <div class="car-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.35rem; font-weight: 700; color: #ffffff;">
                        {b_name} {m_name} <span style="color: #38bdf8;">{y_val}</span>
                    </span>
                    <div>{badge_html}</div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">السعر المعروض:</span><br>
                        <strong style="color: #ffffff; font-size: 1.25rem;">{price_val:,.0f} EGP</strong>
                    </div>
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">السعر العادل المقدر:</span><br>
                        <strong style="color: #38bdf8; font-size: 1.25rem;">{fair_val:,.0f} EGP</strong>
                    </div>
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">المحافظة:</span><br>
                        <span style="color: #cbd5e1; font-size: 1.05rem;">📍 {loc_val}</span>
                    </div>
                    <div>
                        <a href="{item_link}" target="_blank" style="display: inline-block; background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: #38bdf8; padding: 8px 18px; border-radius: 8px; text-decoration: none; font-weight: 700; font-size: 0.95rem;">
                            معاينة الإعلان ↗
                        </a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; color: #64748b; margin-top: 30px;">
        <p style="font-size: 1.05rem;">اكتب مواصفات العربية واضغط <strong>Enter</strong> مباشرة، أو اضغط على أيقونة الكاميرا 📷 لرفع صورة عربية</p>
    </div>
    """, unsafe_allow_html=True)
