import os
import re
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

# ==============================================================================
# CSS: تصميم Premium داكن مستوحى من تصوير السيارات الاحترافي (Black & White)
# ==============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Cairo', 'Plus Jakarta Sans', sans-serif; }

    /* خلفية سينمائية مستوحاة من استوديو تصوير السيارات: إضاءة علوية + عمق */
    .stApp {
        background:
            radial-gradient(60% 45% at 50% 0%, rgba(56, 189, 248, 0.14) 0%, rgba(56, 189, 248, 0) 70%),
            radial-gradient(90% 60% at 50% 110%, rgba(129, 140, 248, 0.10) 0%, rgba(2, 5, 9, 0) 60%),
            linear-gradient(180deg, #05080f 0%, #03060b 45%, #010204 100%);
        color: #f8fafc;
    }

    #MainMenu, footer, header { visibility: hidden; }

    /* شريط علوي رفيع بهوية العلامة */
    .brand-strip {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 10px 0 0 0;
        letter-spacing: 3px;
        font-size: 0.72rem;
        color: #64748b;
        text-transform: uppercase;
        font-weight: 600;
    }
    .brand-strip span.dot { width: 5px; height: 5px; border-radius: 50%; background: #38bdf8; box-shadow: 0 0 8px #38bdf8; }

    /* العنوان الرئيسي */
    .hero-box { text-align: center; padding: 30px 15px 34px 15px; }
    .hero-title {
        font-size: 3.4rem;
        font-weight: 900;
        letter-spacing: -1px;
        background: linear-gradient(100deg, #f8fafc 15%, #38bdf8 55%, #818cf8 80%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
        line-height: 1.1;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.08rem;
        font-weight: 500;
        max-width: 560px;
        margin: 0 auto;
    }
    .hero-underline {
        width: 64px; height: 3px; margin: 18px auto 0 auto;
        background: linear-gradient(90deg, #38bdf8, #c084fc);
        border-radius: 4px;
    }

    /* شريط البحث الموحد - زجاجي فاخر */
    [data-testid="stTextInput"] input {
        background: rgba(10, 16, 28, 0.72) !important;
        backdrop-filter: blur(14px) saturate(140%);
        border: 1.5px solid rgba(148, 163, 184, 0.22) !important;
        color: #ffffff !important;
        border-radius: 46px !important;
        height: 64px !important;
        padding-left: 26px !important;
        padding-right: 68px !important;
        font-size: 1.08rem !important;
        font-weight: 500;
        box-shadow: 0 20px 45px -12px rgba(0, 0, 0, 0.75), inset 0 1px 0 rgba(255,255,255,0.04) !important;
        transition: all 0.25s ease !important;
        direction: rtl;
        text-align: right;
    }
    [data-testid="stTextInput"] input::placeholder { color: #64748b !important; font-weight: 400; }
    [data-testid="stTextInput"] input:focus {
        border-color: rgba(56, 189, 248, 0.65) !important;
        box-shadow: 0 0 0 4px rgba(56, 189, 248, 0.12), 0 20px 45px -12px rgba(0,0,0,0.8) !important;
    }

    /* رفع زرار الصورة ليدخل جوه شريط البحث كأيقونة كاميرا نيون */
    [data-testid="stFileUploader"] {
        margin-top: -64px !important;
        height: 64px !important;
        display: flex !important;
        justify-content: flex-end !important;
        align-items: center !important;
        padding-right: 20px !important;
        pointer-events: none !important;
        border: none !important;
        background: transparent !important;
    }
    [data-testid="stFileUploader"] section {
        padding: 0 !important;
        min-height: unset !important;
        border: none !important;
        background: transparent !important;
        pointer-events: auto !important;
    }
    [data-testid="stFileUploaderDropzone"] { padding: 0 !important; border: none !important; background: transparent !important; }
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzone"] > div:not(:has(button)) { display: none !important; }

    [data-testid="stFileUploader"] button {
        background: rgba(56, 189, 248, 0.10) !important;
        border: 1px solid rgba(56, 189, 248, 0.35) !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        box-shadow: none !important;
        padding: 0 !important;
        cursor: pointer !important;
        color: #38bdf8 !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stFileUploader"] button:hover {
        transform: scale(1.1);
        background: rgba(56, 189, 248, 0.2) !important;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.45) !important;
    }
    [data-testid="stFileUploader"] button::before { content: "📷"; font-size: 1.25rem; }
    [data-testid="stFileUploader"] button span,
    [data-testid="stFileUploader"] button p { display: none !important; }

    /* اسم الملف بعد الرفع */
    [data-testid="stFileUploaderFile"] {
        margin-top: 14px !important;
        background: rgba(10, 16, 28, 0.85) !important;
        border: 1px solid rgba(56, 189, 248, 0.28) !important;
        border-radius: 12px !important;
        pointer-events: auto !important;
    }

    /* عنوان النتائج */
    .results-heading {
        display: flex; align-items: center; gap: 10px;
        font-size: 1.15rem; font-weight: 700; color: #e2e8f0;
        margin: 6px 0 18px 0;
    }
    .results-heading .bar { width: 4px; height: 20px; background: linear-gradient(180deg, #38bdf8, #818cf8); border-radius: 4px; }

    /* شارة التعرف على السيارة بالصورة */
    .vision-badge-wrap { text-align: center; margin: 10px 0 24px 0; }
    .vision-badge {
        display: inline-flex; align-items: center; gap: 8px;
        background: linear-gradient(135deg, rgba(56,189,248,0.14), rgba(129,140,248,0.10));
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: #bae6fd; padding: 8px 20px; border-radius: 30px; font-size: 0.95rem; font-weight: 600;
    }

    /* كروت النتائج - فخمة مع عمق */
    .car-card {
        position: relative;
        background: linear-gradient(155deg, rgba(15, 23, 42, 0.85), rgba(8, 12, 22, 0.85));
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        padding: 24px 26px;
        margin-bottom: 18px;
        overflow: hidden;
        transition: all 0.25s cubic-bezier(.2,.8,.2,1);
        box-shadow: 0 14px 30px -10px rgba(0, 0, 0, 0.6);
    }
    .car-card::before {
        content: "";
        position: absolute; inset: 0;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        opacity: 0; transition: opacity 0.25s ease;
        -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        padding: 1.5px; border-radius: 18px;
        -webkit-mask-composite: xor; mask-composite: exclude;
        pointer-events: none;
    }
    .car-card:hover { transform: translateY(-3px); box-shadow: 0 20px 45px -12px rgba(56, 189, 248, 0.18); }
    .car-card:hover::before { opacity: 0.55; }

    .car-title { font-size: 1.4rem; font-weight: 800; color: #ffffff; }
    .car-title .year { color: #38bdf8; font-weight: 800; }

    .spec-label { color: #64748b; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.3px; }
    .spec-value-main { color: #ffffff; font-size: 1.3rem; font-weight: 800; }
    .spec-value-alt { color: #38bdf8; font-size: 1.3rem; font-weight: 800; }
    .spec-value-loc { color: #cbd5e1; font-size: 1.05rem; font-weight: 600; }

    .cta-link {
        display: inline-block;
        background: linear-gradient(135deg, rgba(56,189,248,0.16), rgba(129,140,248,0.12));
        border: 1px solid rgba(56, 189, 248, 0.5);
        color: #7dd3fc;
        padding: 9px 20px;
        border-radius: 10px;
        text-decoration: none;
        font-weight: 700;
        font-size: 0.92rem;
        transition: all 0.2s ease;
    }
    .cta-link:hover { background: rgba(56, 189, 248, 0.28); color: #ffffff; }

    .deal-badge-great {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.55);
        color: #4ade80;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    .deal-badge-overpriced {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.55);
        color: #f87171;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
    }
    .deal-badge-fair {
        background: rgba(234, 179, 8, 0.12);
        border: 1px solid rgba(234, 179, 8, 0.55);
        color: #facc15;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
    }

    /* حالة عدم وجود بحث */
    .empty-state {
        text-align: center;
        color: #64748b;
        margin-top: 40px;
        padding: 34px 20px;
        border: 1px dashed rgba(148, 163, 184, 0.2);
        border-radius: 18px;
        max-width: 640px;
        margin-left: auto; margin-right: auto;
    }
    .empty-state p { font-size: 1.02rem; margin: 0; }
    .empty-state .icon { font-size: 1.6rem; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

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
<div class="brand-strip"><span class="dot"></span> APEX MOTORS INTELLIGENCE <span class="dot"></span></div>
<div class="hero-box">
    <div class="hero-title">Apex Motors</div>
    <div class="hero-subtitle">محرك البحث الذكي وتقييم أسعار السيارات في السوق المصري</div>
    <div class="hero-underline"></div>
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
            <div class="vision-badge-wrap">
                <span class="vision-badge">📷 تم التعرف على السيارة: <strong>{top_car_label}</strong> ({preds[0]['confidence']*100:.1f}%)</span>
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
        st.markdown('<div class="results-heading"><span class="bar"></span> أفضل النتائج المطابقة</div>', unsafe_allow_html=True)

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
                <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                    <span class="car-title">{b_name} {m_name} <span class="year">{y_val}</span></span>
                    <div>{badge_html}</div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 16px; flex-wrap: wrap; gap: 14px;">
                    <div>
                        <span class="spec-label">السعر المعروض</span><br>
                        <span class="spec-value-main">{price_val:,.0f} EGP</span>
                    </div>
                    <div>
                        <span class="spec-label">السعر العادل المقدر</span><br>
                        <span class="spec-value-alt">{fair_val:,.0f} EGP</span>
                    </div>
                    <div>
                        <span class="spec-label">المحافظة</span><br>
                        <span class="spec-value-loc">📍 {loc_val}</span>
                    </div>
                    <div>
                        <a href="{item_link}" target="_blank" class="cta-link">معاينة الإعلان ↗</a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="empty-state">
        <div class="icon">🔎</div>
        <p>اكتب مواصفات العربية واضغط <strong>Enter</strong> مباشرة، أو اضغط على أيقونة الكاميرا 📷 لرفع صورة عربية</p>
    </div>
    """, unsafe_allow_html=True)
