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
    page_title="Apex Motors | Smart Car Market",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# الصور: الأولى للـHero، والثانية كخلفية كاملة للصفحة
@st.cache_data(show_spinner=False)
def get_image_data(paths, mime):
    for image_path in paths:
        if os.path.exists(image_path):
            with open(image_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime};base64,{encoded}"
    return ""

BG_IMAGE = get_image_data(
    ["background_car.png", "Untitled design.png", "hero_background.png"],
    "image/png"
)

# CSS: شريط بحث مدمج في المنتصف مع أيقونة الكاميرا بالداخل
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{--red:#ff3434;--white:#f7f7f7;--muted:#a9adb5;}
html,body,[data-testid="stAppViewContainer"]{background:#050607!important;}
.stApp{min-height:100vh;background:transparent!important;color:var(--white);font-family:'Inter',sans-serif;}
/* Full-page car background */
.background-car{position:fixed;inset:0;z-index:0;pointer-events:none;background-image:url("{BG_IMAGE}");background-size:cover;background-position:center;background-repeat:no-repeat;opacity:.82;}
.background-car:before{content:"";position:absolute;inset:0;background:linear-gradient(90deg,rgba(0,0,0,.72) 0%,rgba(0,0,0,.34) 48%,rgba(0,0,0,.58) 100%),linear-gradient(180deg,rgba(0,0,0,.30) 0%,rgba(0,0,0,.08) 46%,rgba(0,0,0,.72) 100%);}
.background-car:after{content:"";position:absolute;inset:0;background:radial-gradient(circle at 50% 38%,rgba(255,52,52,.06),transparent 34%);}
.main .block-container{position:relative;z-index:2;max-width:1180px;padding-top:0;padding-bottom:3rem;}
/* Minimal top brand */
.top-brand{display:flex;align-items:center;justify-content:space-between;padding:20px 8px 0;margin-bottom:12px;}
.brand{font-size:1.45rem;font-weight:800;letter-spacing:-.8px;color:#fff;}
.brand span{color:var(--red);}
.brand-mark{color:var(--red);font-size:1.1rem;margin-right:8px;}
.nav{display:flex;gap:32px;color:#d5d7dc;font-size:.82rem;font-weight:600;}
.nav .active{color:#fff;position:relative;}.nav .active:after{content:"";position:absolute;left:0;right:0;bottom:-10px;height:2px;background:var(--red);border-radius:10px;}
/* Hero: no foreground image, only the page background */
.hero-box{position:relative;min-height:320px;display:flex;align-items:center;justify-content:center;text-align:center;margin:0 auto;overflow:hidden;background:transparent;border:0;}
.hero-box:before{content:"";position:absolute;width:520px;height:520px;left:50%;top:50%;transform:translate(-50%,-48%);background:radial-gradient(circle,rgba(0,0,0,.42) 0%,rgba(0,0,0,.20) 42%,transparent 72%);z-index:0;}
.hero-content{position:relative;z-index:2;width:100%;display:flex;flex-direction:column;align-items:center;}
.hero-kicker{display:inline-flex;align-items:center;gap:8px;padding:9px 16px;margin-bottom:12px;border:1px solid rgba(255,255,255,.22);border-radius:999px;background:rgba(0,0,0,.34);color:#e9e9eb;font-size:.67rem;font-weight:800;letter-spacing:3px;backdrop-filter:blur(9px);}
.hero-title{margin:0;color:#fff;font-size:clamp(3.3rem,6vw,5.4rem);line-height:.9;font-weight:800;letter-spacing:-6px;text-shadow:0 10px 40px rgba(0,0,0,.9);}
.hero-title span{color:var(--red);}
.hero-subtitle{color:#e1e3e7;font-size:1.02rem;max-width:620px;margin:12px auto 0;line-height:1.55;text-shadow:0 3px 18px #000;}
.hero-line{width:75px;height:3px;background:var(--red);border-radius:99px;margin:12px auto 0;box-shadow:0 0 22px rgba(255,52,52,.45);}
/* Search */
.search-wrap{position:relative;z-index:10;margin:-2px auto 0;width:min(620px,100%);}
[data-testid="stTextInput"]{position:relative;z-index:10;width:100%;margin:0 auto!important;padding:5px!important;border:1px solid rgba(255,255,255,.23)!important;border-radius:999px!important;background:rgba(10,12,14,.70)!important;box-shadow:0 18px 55px rgba(0,0,0,.60)!important;backdrop-filter:blur(15px);}
[data-testid="stTextInput"]:before{content:"";position:absolute;top:-1px;left:25%;width:50%;height:2px;border-radius:99px;background:linear-gradient(90deg,transparent,#fff,var(--red),#fff,transparent);}
[data-testid="stTextInput"] input{background:rgba(255,255,255,.035)!important;border:1px solid rgba(255,255,255,.12)!important;color:#fff!important;border-radius:999px!important;height:46px!important;padding:0 54px 0 20px!important;font-size:1rem!important;direction:ltr;text-align:left;}
[data-testid="stTextInput"] input::placeholder{color:#b9bcc2!important;}
[data-testid="stTextInput"] input:focus{border-color:rgba(255,52,52,.8)!important;box-shadow:0 0 0 3px rgba(255,52,52,.07),0 0 25px rgba(255,52,52,.14)!important;}
/* Camera uploader */
[data-testid="stFileUploader"]{margin-top:-46px!important;height:46px!important;display:flex!important;justify-content:flex-end!important;align-items:center!important;padding-right:8px!important;pointer-events:none!important;border:none!important;background:transparent!important;position:relative;z-index:20;}
[data-testid="stFileUploader"] section{padding:0!important;min-height:unset!important;border:none!important;background:transparent!important;pointer-events:auto!important;}
[data-testid="stFileUploaderDropzone"]{padding:0!important;border:none!important;background:transparent!important;}
[data-testid="stFileUploaderDropzoneInstructions"],[data-testid="stFileUploaderDropzone"]>div:not(:has(button)){display:none!important;}
[data-testid="stFileUploader"] button{width:34px!important;height:34px!important;border-radius:50%!important;background:rgba(15,15,17,.88)!important;border:1px solid rgba(255,52,52,.55)!important;padding:4px!important;color:#fff!important;transition:.2s ease!important;}
[data-testid="stFileUploader"] button:hover{transform:scale(1.08);background:rgba(255,52,52,.14)!important;border-color:var(--red)!important;}
[data-testid="stFileUploader"] button:before{content:"📷";font-size:.88rem;}
[data-testid="stFileUploader"] button span,[data-testid="stFileUploader"] button p{display:none!important;}
[data-testid="stFileUploaderFile"]{margin-top:14px!important;background:rgba(5,5,6,.90)!important;border:1px solid rgba(255,255,255,.15)!important;border-radius:12px!important;}
/* Results */
.results-title{color:#fff;font-size:1.15rem;font-weight:700;margin:30px 0 18px;text-align:left;}
.car-card{background:rgba(5,6,8,.78);border:1px solid rgba(255,255,255,.15);border-radius:18px;padding:22px;margin-bottom:16px;box-shadow:0 16px 40px rgba(0,0,0,.52);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);}
.car-card:hover{border-color:rgba(255,255,255,.30);transform:translateY(-2px);}
.deal-badge-great{background:rgba(34,197,94,.13);border:1px solid rgba(34,197,94,.65);color:#86efac;font-weight:700;padding:4px 12px;border-radius:20px;font-size:.85rem;}
.deal-badge-overpriced{background:rgba(239,68,68,.13);border:1px solid rgba(239,68,68,.65);color:#fca5a5;font-weight:700;padding:4px 12px;border-radius:20px;font-size:.85rem;}
.deal-badge-fair{background:rgba(234,179,8,.13);border:1px solid rgba(234,179,8,.65);color:#fde047;font-weight:700;padding:4px 12px;border-radius:20px;font-size:.85rem;}
.empty-state{text-align:center;color:#b8bcc4;margin-top:36px;text-shadow:0 2px 12px #000;}
@media(max-width:700px){.hero-box{min-height:310px}.hero-title{font-size:3.1rem;letter-spacing:-2px}.hero-subtitle{font-size:.86rem;max-width:90%}.search-wrap{width:94%}.background-car{background-position:center;opacity:.72;}}
</style>
""".replace("{BG_IMAGE}", BG_IMAGE), unsafe_allow_html=True)

# صورة الخلفية فوق طبقة التطبيق وتحت كل عناصر الواجهة
st.markdown('<div class="background-car"></div>', unsafe_allow_html=True)

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
    st.error(f"⚠️ Project loading error: {e}")
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
    <div class="hero-content">
        <div class="hero-kicker">✦ SMART CAR MARKET</div>
        <div class="hero-title">Apex <span>Motors</span></div>
        <div class="hero-line"></div>
        <div class="hero-subtitle">Find the right car, get expert insights, and make smarter decisions with the power of AI.</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ضبط الشريط في منتصف الشاشة بدقة
_, col_search, _ = st.columns([1, 2.6, 1])

with col_search:
    user_query = st.text_input(
        "Search",
        placeholder="Type your car requirements and press Enter...",
        label_visibility="collapsed"
    )
    uploaded_file = st.file_uploader(
        "Upload car image",
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
                    📷 Detected car: <strong>{top_car_label}</strong> ({preds[0]['confidence']*100:.1f}%)
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
        st.write("### Best Matches")

        for _, row in top_results.iterrows():
            b_name = row.get("brand", "")
            m_name = row.get("model", "")
            y_val = int(row.get("year", 0)) if pd.notna(row.get("year")) and row.get("year") != 0 else ""
            price_val = int(row.get("price", 0)) if pd.notna(row.get("price")) else 0
            fair_val = int(row.get("predicted_fair_price", 0)) if pd.notna(row.get("predicted_fair_price")) else 0
            deal_tag = row.get("deal_label", "Fair Price")
            loc_val = row.get("location", "Egypt")
            item_link = row.get("item_url", "#")

            if deal_tag == "Great Deal":
                badge_html = '<span class="deal-badge-great">🟢 Great Deal</span>'
            elif deal_tag == "Overpriced":
                badge_html = '<span class="deal-badge-overpriced">🔴 Overpriced</span>'
            else:
                badge_html = '<span class="deal-badge-fair">🟡 Fair Price</span>'

            st.markdown(f"""
            <div class="car-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.35rem; font-weight: 700; color: #ffffff;">
                        {b_name} {m_name} <span style="color: var(--red);">{y_val}</span>
                    </span>
                    <div>{badge_html}</div>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 12px; flex-wrap: wrap; gap: 10px;">
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">Listed Price:</span><br>
                        <strong style="color: #ffffff; font-size: 1.25rem;">{price_val:,.0f} EGP</strong>
                    </div>
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">Estimated Fair Price:</span><br>
                        <strong style="color: var(--red); font-size: 1.25rem;">{fair_val:,.0f} EGP</strong>
                    </div>
                    <div>
                        <span style="color: #94a3b8; font-size: 0.85rem;">Location:</span><br>
                        <span style="color: #cbd5e1; font-size: 1.05rem;">📍 {loc_val}</span>
                    </div>
                    <div>
                        <a href="{item_link}" target="_blank" style="display: inline-block; background: rgba(56, 189, 248, 0.15); border: 1px solid #38bdf8; color: var(--red); padding: 8px 18px; border-radius: 8px; text-decoration: none; font-weight: 700; font-size: 0.95rem;">
                            View Listing ↗
                        </a>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div style="text-align: center; color: #64748b; margin-top: 30px;">
        <p style="font-size: 1.05rem;">Type your car requirements and press <strong>Enter</strong>, or click the camera 📷 to upload a car image</p>
    </div>
    """, unsafe_allow_html=True)
