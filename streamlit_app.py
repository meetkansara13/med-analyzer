import streamlit as st
import httpx
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import mlflow
import re
from mlflow.tracking import MlflowClient
from datetime import datetime
import pypdf

API_URL = "http://127.0.0.1:801"
MLFLOW_URI = "http://localhost:5000"

st.set_page_config(
    page_title="MedAnalyzer AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg:       #0d1117;
    --bg2:      #131a24;
    --bg3:      #182030;
    --panel:    #1a2436;
    --stroke:   rgba(99,165,255,0.14);
    --stroke2:  rgba(99,165,255,0.26);
    --text:     #e8f1ff;
    --muted:    #7a95b8;
    --cyan:     #29d9c2;
    --blue:     #3b82f6;
    --violet:   #8b5cf6;
    --green:    #22d3a5;
    --amber:    #f59e0b;
    --red:      #ef4444;
    --pink:     #ec4899;
    --r:        16px;
    --r2:       12px;
    --shadow:   0 4px 24px rgba(0,0,0,0.32);
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }

/* ── APP BG ── */
.stApp {
    background: var(--bg);
    color: var(--text);
}
.main .block-container { padding: 0 1.4rem 2.5rem; max-width: 1480px; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--stroke);
    width: 220px !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.4rem 0.9rem; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2);
    border: 1px solid var(--stroke);
    border-radius: 14px;
    padding: 5px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    color: var(--muted);
    font-weight: 600;
    font-size: 0.86rem;
    padding: 0.65rem 1rem;
    transition: all 0.2s;
}
.stTabs [data-baseweb="tab"]:hover { background: rgba(59,130,246,0.1); color: #c8dcff; }
.stTabs [aria-selected="true"] {
    background: var(--blue) !important;
    color: white !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
}

/* ── CARDS ── */
.card {
    background: var(--panel);
    border: 1px solid var(--stroke);
    border-radius: var(--r);
    padding: 1.1rem 1.15rem;
    margin-bottom: 0.9rem;
}
.card-blue   { border-color: rgba(59,130,246,0.3);  box-shadow: 0 0 0 1px rgba(59,130,246,0.08); }
.card-cyan   { border-color: rgba(41,217,194,0.28); box-shadow: 0 0 0 1px rgba(41,217,194,0.07); }
.card-green  { border-color: rgba(34,211,165,0.28); box-shadow: 0 0 0 1px rgba(34,211,165,0.07); }
.card-amber  { border-color: rgba(245,158,11,0.28); box-shadow: 0 0 0 1px rgba(245,158,11,0.07); }
.card-red    { border-color: rgba(239,68,68,0.28);  box-shadow: 0 0 0 1px rgba(239,68,68,0.07); }
.card-violet { border-color: rgba(139,92,246,0.28); box-shadow: 0 0 0 1px rgba(139,92,246,0.07); }

/* keep old names mapping */
.card-accent  { border-color: rgba(59,130,246,0.3);  box-shadow: 0 0 0 1px rgba(59,130,246,0.08); }
.card-success { border-color: rgba(34,211,165,0.28); }
.card-warning { border-color: rgba(245,158,11,0.28); }
.card-danger  { border-color: rgba(239,68,68,0.28);  }

/* ── STAT PILL ── */
.stat-pill {
    display: flex;
    align-items: center;
    gap: 14px;
    background: var(--panel);
    border: 1px solid var(--stroke);
    border-radius: var(--r);
    padding: 1rem 1.2rem;
}
.stat-icon {
    width: 46px; height: 46px;
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.stat-val { font-size: 1.45rem; font-weight: 800; line-height: 1; }
.stat-lbl { font-size: 0.72rem; color: var(--muted); margin-top: 3px; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── METRIC GRID ── */
.metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 0.8rem 0; }
.metric-box {
    background: var(--panel);
    border: 1px solid var(--stroke);
    border-radius: var(--r2);
    padding: 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-box::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg,var(--blue),var(--cyan));
}
.metric-num { font-size: 1.85rem; font-weight: 800; line-height:1.1; font-family:'JetBrains Mono',monospace; }
.metric-lbl { font-size: 0.68rem; color: var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-top:4px; }
.blue   { color: var(--blue); }
.green  { color: var(--green); }
.orange { color: var(--amber); }
.purple { color: var(--violet); }
.teal   { color: var(--cyan); }

/* ── ENTITY TAGS ── */
.tag { display:inline-block; padding:3px 10px; border-radius:999px; font-size:0.75rem; margin:3px; font-family:'JetBrains Mono',monospace; font-weight:600; }
.t-disease   { background:rgba(239,68,68,0.1);   color:#fca5a5; border:1px solid rgba(239,68,68,0.25); }
.t-drug      { background:rgba(59,130,246,0.1);  color:#93c5fd; border:1px solid rgba(59,130,246,0.25); }
.t-symptom   { background:rgba(245,158,11,0.1);  color:#fcd34d; border:1px solid rgba(245,158,11,0.25); }
.t-anatomy   { background:rgba(34,211,165,0.1);  color:#6ee7cc; border:1px solid rgba(34,211,165,0.25); }
.t-procedure { background:rgba(139,92,246,0.1);  color:#c4b5fd; border:1px solid rgba(139,92,246,0.25); }
.t-lab       { background:rgba(41,217,194,0.1);  color:#6ee7d5; border:1px solid rgba(41,217,194,0.25); }
.t-other     { background:rgba(122,149,184,0.1); color:#a8bdd6; border:1px solid rgba(122,149,184,0.2); }

/* ── CHAT BUBBLES ── */
.bubble-user { background:var(--blue); color:white; border-radius:18px 18px 6px 18px; padding:12px 16px; margin:8px 0 8px 14%; line-height:1.6; font-size:0.9rem; }
.bubble-bot  { background:var(--panel); border:1px solid var(--stroke); border-radius:18px 18px 18px 6px; padding:14px 18px; margin:8px 14% 8px 0; line-height:1.7; font-size:0.9rem; }
.bubble-meta { color:var(--muted); font-size:0.74rem; margin-top:7px; }

/* ── SECTION TITLE ── */
.sec-title {
    font-size: 0.7rem; font-weight: 700; color: var(--muted);
    letter-spacing: 0.14em; text-transform: uppercase;
    margin: 1.3rem 0 0.7rem;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--stroke);
    font-family: 'JetBrains Mono', monospace;
}

/* ── REPORT LABELS ── */
.report-label { font-size:0.72rem; font-weight:600; color:var(--muted); text-transform:uppercase; letter-spacing:0.08em; }
.report-value { font-size:0.93rem; color:var(--text); margin-top:2px; }
.report-highlight { color:var(--green); font-weight:600; }
.report-critical  { color:#fca5a5; font-weight:600; }
.report-warn      { color:#fcd34d; font-weight:600; }

/* ── BUTTONS ── */
.stButton>button {
    background: var(--panel) !important;
    color: var(--text) !important;
    border: 1px solid var(--stroke2) !important;
    border-radius: 10px !important;
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    padding: 0.65rem 1rem !important;
    transition: all 0.18s !important;
}
.stButton>button:hover {
    background: rgba(59,130,246,0.15) !important;
    border-color: rgba(59,130,246,0.45) !important;
    color: #c8dcff !important;
    transform: translateY(-1px) !important;
}
button[kind="primary"] {
    background: var(--blue) !important;
    border-color: transparent !important;
    color: white !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 14px rgba(59,130,246,0.35) !important;
}
button[kind="primary"]:hover {
    background: #2563eb !important;
    color: white !important;
}

/* ── INPUTS ── */
.stTextArea textarea, .stTextInput input {
    background: var(--bg3) !important;
    color: var(--text) !important;
    border: 1px solid var(--stroke2) !important;
    border-radius: 12px !important;
    font-family: 'Inter',sans-serif !important;
}
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important;
}
.stSelectbox [data-baseweb="select"] { background: var(--bg3) !important; border: 1px solid var(--stroke2) !important; border-radius: 10px !important; }

/* ── STATUS DOTS ── */
.status-dot { height:8px; width:8px; border-radius:50%; display:inline-block; margin-right:5px; }
.dot-green { background:var(--green); box-shadow:0 0 8px var(--green); }
.dot-red   { background:var(--red);   box-shadow:0 0 8px var(--red); }

/* ── EXPANDER ── */
div[data-testid="stExpander"] { background:var(--panel); border:1px solid var(--stroke); border-radius:12px; }

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border:1px solid var(--stroke); border-radius:12px; overflow:hidden; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] { background:var(--bg3) !important; border:2px dashed rgba(59,130,246,0.25) !important; border-radius:14px !important; }

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:rgba(59,130,246,0.3); border-radius:3px; }

/* ── WELCOME BANNER ── */
.welcome-banner {
    background: linear-gradient(135deg, #0f2044 0%, #1a3a5c 50%, #0d2a3a 100%);
    border: 1px solid rgba(59,130,246,0.2);
    border-radius: 20px;
    padding: 1.4rem 1.6rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 1rem;
}
.welcome-banner::after {
    content:'';
    position:absolute; right:-60px; top:-60px;
    width:240px; height:240px;
    background: radial-gradient(circle, rgba(41,217,194,0.18), transparent 60%);
    pointer-events:none;
}
.welcome-title { font-size:1.55rem; font-weight:800; color:#f0f8ff; margin-bottom:4px; }
.welcome-sub   { font-size:0.9rem;  color:#9ab4cc; line-height:1.7; max-width:560px; }

/* ── NAV ITEM ── */
.nav-item {
    display:flex; align-items:center; gap:10px;
    padding: 10px 12px;
    border-radius: 10px;
    font-size:0.88rem; font-weight:600;
    color: var(--muted);
    cursor:pointer;
    transition: all 0.15s;
    margin-bottom:4px;
    text-decoration:none;
}
.nav-item:hover { background:rgba(59,130,246,0.1); color:#c8dcff; }
.nav-item.active { background:var(--blue); color:white; box-shadow:0 3px 12px rgba(59,130,246,0.3); }

/* ── PROGRESS BAR ── */
.prog-wrap { height:6px; background:rgba(255,255,255,0.08); border-radius:3px; overflow:hidden; margin-top:8px; }
.prog-fill  { height:100%; border-radius:3px; transition:width 0.8s ease; }

/* ── HABIT CARD ── */
.habit-card {
    background: var(--bg3);
    border: 1px solid var(--stroke);
    border-radius: 14px;
    padding: 14px 16px;
    text-align:center;
}
.habit-icon {
    width:48px; height:48px;
    border-radius:14px;
    display:flex; align-items:center; justify-content:center;
    font-size:22px;
    margin: 0 auto 8px;
}
.habit-val  { font-size:1.1rem; font-weight:800; color:var(--text); }
.habit-lbl  { font-size:0.72rem; color:var(--muted); margin-top:2px; }
.habit-prog-lbl { font-size:0.7rem; color:var(--muted); margin-top:8px; }

/* ── MOOD SELECTOR ── */
.mood-row { display:flex; gap:8px; justify-content:space-between; }
.mood-btn {
    width:38px; height:38px;
    border-radius:10px;
    background:var(--bg3);
    border:1px solid var(--stroke);
    display:flex; align-items:center; justify-content:center;
    font-size:20px;
    cursor:pointer;
    transition:all 0.15s;
}
.mood-btn:hover, .mood-btn.active {
    background:rgba(59,130,246,0.15);
    border-color:var(--blue);
    transform:scale(1.12);
}

/* ── WELLNESS WAVE ── */
.wave-label { font-size:0.7rem; color:var(--muted); font-family:'JetBrains Mono',monospace; }

/* ── GOAL GAUGE ── */
.gauge-wrap { text-align:center; }
.gauge-big  { font-size:2.4rem; font-weight:800; }
.gauge-sub  { font-size:0.82rem; color:var(--muted); line-height:1.5; margin-top:4px; }

/* ── PLAN STYLES ── */
.plan-pill { display:inline-flex; align-items:center; padding:4px 12px; border-radius:999px; background:rgba(59,130,246,0.1); border:1px solid rgba(59,130,246,0.22); color:#93c5fd; font-size:0.76rem; font-weight:600; margin:3px 4px 3px 0; }
.plan-step { display:flex; gap:10px; align-items:flex-start; padding:8px 0; border-bottom:1px solid var(--stroke); }
.plan-step:last-child { border-bottom:none; }
.plan-step-num { width:28px; height:28px; border-radius:8px; background:rgba(59,130,246,0.15); display:flex; align-items:center; justify-content:center; color:#93c5fd; font-weight:800; font-size:0.8rem; flex-shrink:0; }
.plan-step-text { color:#d0dff5; line-height:1.7; font-size:0.88rem; }
.plan-item-card { background:var(--bg3); border:1px solid var(--stroke); border-radius:12px; padding:10px 14px; margin-bottom:7px; }
.plan-item-text { color:#ccdaf5; line-height:1.7; font-size:0.87rem; }
.plan-stack     { display:flex; flex-direction:column; gap:0; }
.plan-section-heading { font-size:0.72rem; color:var(--muted); font-weight:700; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:10px; font-family:'JetBrains Mono',monospace; }
.plan-mini-title { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--muted); margin-bottom:6px; font-family:'JetBrains Mono',monospace; }
.plan-big-value  { font-size:1.35rem; font-weight:800; color:#f0f8ff; }
.plan-note       { color:#b0c8e4; line-height:1.75; font-size:0.89rem; }
.timeline-entry  { display:flex; gap:10px; align-items:flex-start; padding:8px 0; border-bottom:1px solid var(--stroke); }
.timeline-entry:last-child { border-bottom:none; }
.timeline-dot    { width:22px; height:22px; border-radius:7px; background:rgba(59,130,246,0.15); display:flex; align-items:center; justify-content:center; color:#93c5fd; font-size:0.74rem; flex-shrink:0; margin-top:2px; }
.timeline-time   { font-size:0.7rem; color:#7a95b8; font-weight:700; margin-bottom:2px; }
.timeline-text   { font-size:0.87rem; color:#d0dff5; line-height:1.6; }
.plan-list       { margin:0; padding-left:1rem; }
.plan-list li    { margin-bottom:7px; line-height:1.7; color:#d0dff5; font-size:0.88rem; }

/* ── INFO GRID ── */
.info-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:0.6rem 0 1rem; }
.info-panel { background:var(--panel); border:1px solid var(--stroke); border-radius:var(--r); padding:1rem 1.1rem; }
.info-panel-label { font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--muted); margin-bottom:6px; font-family:'JetBrains Mono',monospace; }
.info-panel-value { font-size:0.95rem; color:var(--text); line-height:1.75; }

@media(max-width:900px){
    .metric-grid,.info-grid{grid-template-columns:repeat(2,1fr);}
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ──────────────────────────────────────────────
for k, v in [
    ("analysis_result", None), ("chat_history", []),
    ("doc_id", None), ("extracted_text", ""),
    ("pending_question", None), ("doc_summary_for_questions", ""),
    ("doc_entities_for_questions", {}), ("doc_summary_data_for_questions", {}),
    ("q_input", ""), ("auto_send_question", False),
    ("mood", None), ("wellness_scores", [72,68,75,80,74,82,79]),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helpers (unchanged from original) ─────────────────────────
def check_api():
    try:
        r = httpx.get(f"{API_URL}/health", timeout=2)
        return r.status_code == 200
    except:
        return False

def call_analyze(text, doc_type):
    return httpx.post(f"{API_URL}/analyze/text", json={"text": text, "doc_type": doc_type}, timeout=120)

def call_query(question, doc_id):
    return httpx.post(f"{API_URL}/chat/query", json={"question": question, "doc_id": doc_id}, timeout=120)

def classify_value(label, value_str):
    val_l = value_str.lower()
    if any(x in val_l for x in ["↑","↓","high","low","elevated","critical","abnormal"]):
        if any(x in val_l for x in ["critical","panic"]): return "report-critical"
        return "report-warn"
    return "report-highlight"

def humanize_category(category):
    labels = {"diseases":"Health Conditions","drugs":"Medicines","symptoms":"Symptoms","anatomy":"Body Areas","procedures":"Tests & Procedures","lab_values":"Lab Results","other":"Other Terms"}
    return labels.get(category, category.replace("_"," ").title())

def unique_keep_order(items):
    seen, out = set(), []
    for item in items or []:
        t = str(item).strip()
        if t and t.lower() not in seen:
            seen.add(t.lower()); out.append(t)
    return out

def enrich_dashboard_entities(entities, summary_data=None):
    summary_data = summary_data or {}
    enriched = {k: list(v) for k, v in (entities or {}).items()}
    for key in ["diseases","drugs","symptoms","lab_values","procedures","anatomy","other"]:
        enriched.setdefault(key, [])
    other_items = list(enriched.get("other", []))
    summary_text = " ".join([summary_data.get("summary","")] + (summary_data.get("key_findings",[]) or []) + (summary_data.get("recommendations",[]) or []))
    med_suffixes = ("mg","mcg","ml","tablet","tab","capsule","syrup","injection","po","iv","once daily","twice daily")
    symptom_terms = {"pain","fever","cough","breathlessness","shortness of breath","vomiting","nausea","headache","dizziness","fatigue","weakness","palpitations","syncope","swelling","chest pain"}
    disease_terms = {"diabetes","hypertension","asthma","copd","pneumonia","infection","nstemi","stemi","acs","anemia","kidney disease","thyroid","stroke","heart failure","cad","tb","covid","hepatitis","arthritis"}
    procedure_terms = {"ecg","echo","x-ray","xray","ct","mri","ultrasound","scan","angiography","biopsy","test","procedure","examination","impression"}
    lab_terms = {"troponin","hb","hemoglobin","glucose","creatinine","wbc","rbc","platelet","sodium","potassium","bilirubin","sgpt","sgot","hba1c","cholesterol","triglyceride","bun","tsh","ldl","hdl","spo2","bp"}
    dd, ds, di, dl, dp = [], [], [], [], []
    for raw in other_items + [summary_text]:
        for part in re.split(r"[\n,;|]+", raw):
            item = part.strip(" .:-")
            if not item: continue
            il = item.lower()
            if any(t in il for t in med_suffixes) and len(item.split()) <= 12: dd.append(item); continue
            if any(t in il for t in lab_terms) or re.search(r"\b\d+(?:\.\d+)?\s?(mg/dl|g/dl|ng/ml|mmhg|bpm|%)\b", il): dl.append(item); continue
            if any(t in il for t in disease_terms): di.append(item); continue
            if any(t in il for t in symptom_terms): ds.append(item); continue
            if any(t in il for t in procedure_terms): dp.append(item); continue
    enriched["drugs"]      = unique_keep_order(enriched["drugs"]      + dd)
    enriched["symptoms"]   = unique_keep_order(enriched["symptoms"]   + ds)
    enriched["diseases"]   = unique_keep_order(enriched["diseases"]   + di)
    enriched["lab_values"] = unique_keep_order(enriched["lab_values"] + dl)
    enriched["procedures"] = unique_keep_order(enriched["procedures"] + dp)
    return enriched

def is_useful_question_target(text):
    value = (text or "").strip()
    if not value or len(value) < 4: return False
    bad = ["contact","consultation","section","general hospital","medical center","policy no","confidential","patient information","vital signs & anthropometric measurements"]
    if any(p in value.lower() for p in bad): return False
    if re.fullmatch(r"[\d\s/.\-:]+", value): return False
    return True

def summarize_overview(summary, entities):
    overview = []
    d = entities.get("diseases") or []
    dr = entities.get("drugs") or []
    sy = entities.get("symptoms") or []
    la = entities.get("lab_values") or []
    pr = entities.get("procedures") or []
    rec = summary.get("recommendations",[]) or []
    if d:  overview.append(f"Main condition: {d[0]}.")
    if sy: overview.append(f"Symptoms include {', '.join(sy[:3])}.")
    if dr: overview.append(f"Medicines: {', '.join(dr[:3])}.")
    if la: overview.append("Lab findings need attention.")
    if pr: overview.append(f"Tests: {', '.join(pr[:2])}.")
    if rec: overview.append("Follow-up recommendations are present.")
    if not overview:
        return summary.get("summary","") or "Report analyzed with limited structured details."
    return " ".join(overview[:4])

def humanize_run_type(rt): return "Document Analysis" if rt == "analysis" else "Question Answering"
def humanize_run_name(name):
    if not name: return "Unnamed Run"
    c = str(name).replace("_"," ").strip()
    if c.lower().startswith("analysis "): return "Document analysis"
    if c.lower().startswith("query "): return "Question asked"
    return c[:1].upper() + c[1:]

def build_attention_profile(result):
    summary = result.get("summary",{}) or {}
    entities = result.get("entities",{}) or {}
    findings = summary.get("key_findings",[]) or []
    combined = ((summary.get("summary","") or "") + " " + " ".join(findings)).lower()
    score = 0
    if any(w in combined for w in ["critical","urgent","severe","elevated","abnormal","depression"]): score += 2
    if entities.get("lab_values"): score += 1
    if entities.get("diseases"):   score += 1
    if len(findings) >= 3:          score += 1
    if score >= 4:   level, color, note = "High attention",     "#fca5a5", "Findings need timely medical review and close follow-up."
    elif score >= 2: level, color, note = "Moderate attention", "#fcd34d", "Findings should be understood carefully and monitored."
    else:            level, color, note = "Routine attention",  "#6ee7cc", "Appears stable — still review in clinical context."
    focus = []
    if entities.get("diseases"):   focus.append(f"Condition focus: {', '.join(entities['diseases'][:2])}")
    if entities.get("lab_values"): focus.append("Lab monitoring may be important.")
    if entities.get("drugs"):      focus.append(f"Medication review: {', '.join(entities['drugs'][:3])}")
    if summary.get("recommendations",[]): focus.append("Follow-up planning mentioned.")
    if not focus: focus.append("General medical review recommended.")
    return {"level":level,"color":color,"note":note,"focus_areas":focus[:4]}

def get_total_medical_details(entities): return sum(len(v) for v in (entities or {}).values() if v)

def infer_report_mode(result, doc_type):
    entities = result.get("entities",{}) or {}
    summary  = result.get("summary",{}) or {}
    combined = ((summary.get("summary","") or "") + " " + " ".join(summary.get("key_findings",[]) or []) + " " + " ".join(summary.get("recommendations",[]) or [])).lower()
    if entities.get("lab_values") or any(x in combined for x in ["troponin","hb","glucose","creatinine","lab"]): return "Lab-focused report"
    if entities.get("drugs") and any(x in combined for x in ["prescribed","tablet","capsule","medicine","dose"]): return "Treatment / prescription"
    if any(x in combined for x in ["discharge","admission","hospital","follow-up"]): return "Discharge / care transition"
    if any(x in combined for x in ["x-ray","mri","ct","scan","impression"]): return "Imaging / test report"
    return doc_type.replace("_"," ").title() if doc_type else "Medical report"

def build_report_story(result):
    summary = result.get("summary",{}) or {}
    entities = result.get("entities",{}) or {}
    kf = summary.get("key_findings",[]) or []
    rec = summary.get("recommendations",[]) or []
    hl = []
    if entities.get("diseases"):   hl.append(f"Main condition: {entities['diseases'][0]}")
    if entities.get("symptoms"):   hl.append(f"Symptoms: {', '.join(entities['symptoms'][:3])}")
    if entities.get("lab_values"): hl.append("Important lab signals detected.")
    if entities.get("drugs"):      hl.append(f"Medicines: {', '.join(entities['drugs'][:3])}")
    if kf:  hl.append(f"Key concern: {kf[0]}")
    if rec: hl.append(f"Next step: {rec[0]}")
    return hl[:6]

def build_treatment_guidance(result):
    summary = result.get("summary",{}) or {}
    entities = result.get("entities",{}) or {}
    findings = summary.get("key_findings",[]) or []
    recs     = summary.get("recommendations",[]) or []
    combined = (" ".join(findings + recs) + " " + (summary.get("summary","") or "")).lower()
    conditions = entities.get("diseases",[]) or []
    drugs      = entities.get("drugs",[])    or []
    labs       = entities.get("lab_values",[]) or []
    tt, dq, ls = [], [], []
    if drugs:       tt.append(f"Medicines like {', '.join(drugs[:3])} should only be changed based on a doctor's advice.")
    if conditions:  tt.append(f"Treatment for {conditions[0]} depends on severity, test findings, and medical history.")
    if labs:        tt.append("Lab findings may affect treatment — match with doctor recommendations.")
    if any(x in combined for x in ["infection","fever","pneumonia"]): tt.append("Infection suspected — doctor may decide on antibiotics or monitoring.")
    if any(x in combined for x in ["bp","blood pressure","hypertension","heart","chest pain","troponin","nstemi","stemi"]): tt.append("Heart/BP findings need doctor-guided treatment.")
    if not tt: tt.append("Review this report with a doctor before any medicine change.")
    if conditions: dq.append(f"What is confirmed for {conditions[0]}?")
    dq.extend(["What medicines do I need, and for how long?","Any side effects or interactions to watch?","Do I need repeat tests or urgent monitoring?"])
    if any(x in combined for x in ["sugar","glucose","diabetes"]): ls.append("Follow a doctor-approved meal plan and track blood sugar.")
    if any(x in combined for x in ["blood pressure","hypertension","heart","chest pain"]): ls.append("Limit salt, avoid smoking, follow activity advice.")
    if any(x in combined for x in ["infection","fever"]): ls.append("Rest, stay hydrated, watch for worsening symptoms.")
    ls.extend(["Keep a written list of current medicines and doses.","Take follow-up appointments seriously."])
    return {"treatment_topics":tt[:4],"doctor_questions":dq[:4],"lifestyle_support":ls[:4],
            "warning":"This is educational support only — not a prescription. Confirm every medicine decision with your doctor."}

def extract_medicine_schedule(result):
    entities = enrich_dashboard_entities(result.get("entities",{}) or {}, result.get("summary",{}) or {})
    meds = entities.get("drugs",[]) or []
    schedule = {"Morning":{"time":"8:00 AM","items":[]},"Afternoon":{"time":"1:00 PM","items":[]},"Evening":{"time":"6:00 PM","items":[]},"Night":{"time":"9:00 PM","items":[]}}
    for med in meds:
        ml = med.lower()
        if any(x in ml for x in ["od","once daily","morning"]):         schedule["Morning"]["items"].append(med)
        elif any(x in ml for x in ["bd","twice daily"]):                schedule["Morning"]["items"].append(med); schedule["Night"]["items"].append(med)
        elif any(x in ml for x in ["hs","night","bedtime"]):            schedule["Night"]["items"].append(med)
        elif any(x in ml for x in ["afternoon","lunch"]):               schedule["Afternoon"]["items"].append(med)
        elif any(x in ml for x in ["evening"]):                         schedule["Evening"]["items"].append(med)
        else:                                                             schedule["Morning"]["items"].append(med)
    for k in schedule: schedule[k]["items"] = unique_keep_order(schedule[k]["items"])
    return schedule

def build_diet_plan(result):
    summary = result.get("summary",{}) or {}
    text = ((summary.get("summary","") or "") + " " + " ".join(summary.get("key_findings",[]) or []) + " " + " ".join(summary.get("recommendations",[]) or [])).lower()
    bk = ["Light breakfast: oats, upma, poha, or multigrain toast.","Unsweetened tea, milk, or warm water."]
    lu = ["Roti or brown rice with dal, vegetable, and salad.","Keep salt and oil moderate."]
    sn = ["Fruit, roasted chana, sprouts, or a handful of nuts.","Avoid sugary and deep-fried snacks."]
    di = ["Lighter than lunch: soup, vegetables, dal, smaller carbs.","Finish dinner 2–3 hours before sleep."]
    dr = ["Stay well hydrated with water through the day.","Avoid sugary drinks and packaged juices."]
    if any(x in text for x in ["diabetes","blood sugar","glucose","hba1c","diabetic"]): bk.append("High-fiber choices; avoid sweet cereals."); sn.append("Low-glycemic snacks; discuss fruit quantity with doctor.")
    if any(x in text for x in ["heart","nstemi","stemi","cholesterol","cardiac","hypertension","blood pressure"]): lu.append("Heart-friendly plate: more vegetables, less fried food."); di.append("Avoid heavy oily dinners."); dr.append("Avoid salty soups, sodas, and energy drinks.")
    if any(x in text for x in ["kidney","creatinine"]): dr.append("Fluid intake should follow doctor's guidance.")
    return {"Breakfast":unique_keep_order(bk)[:3],"Lunch":unique_keep_order(lu)[:3],"Healthy Snacks":unique_keep_order(sn)[:3],"Dinner":unique_keep_order(di)[:3],"Drinks":unique_keep_order(dr)[:3]}

def build_exercise_plan(result):
    summary = result.get("summary",{}) or {}
    text = ((summary.get("summary","") or "") + " " + " ".join(summary.get("key_findings",[]) or []) + " " + " ".join(summary.get("recommendations",[]) or [])).lower()
    plans = []
    if any(x in text for x in ["no heavy lifting","cardiac","nstemi","stemi","chest pain"]):
        plans.append({"title":"Gentle Recovery Walk","steps":["Warm up with slow breathing for 2 min.","Walk slowly 5–10 min on flat ground.","Stop if chest pain, dizziness, or breathlessness increases.","Increase duration only with doctor approval."]})
    if any(x in text for x in ["body pain","muscle","weakness","fatigue"]):
        plans.append({"title":"Light Mobility Routine","steps":["Neck turns and shoulder rolls — 1 min.","Gentle arm raises and ankle rotations — 2 min.","Sit-to-stand slowly 5 times if comfortable.","Seated deep breathing — 2 min."]})
    if any(x in text for x in ["weight","diabetes","blood sugar","hypertension"]):
        plans.append({"title":"Daily Metabolic Activity","steps":["10–15 min walk after main meal (if cleared).","One consistent daily activity time.","Don't skip for many days in a row.","Track how you feel before and after."]})
    if not plans:
        plans.append({"title":"Safe Daily Movement","steps":["5–10 minutes of light walking.","Gentle stretching for shoulders, legs, back.","Don't push through pain or dizziness.","Ask your doctor before any intense workout."]})
    return plans[:3]

def build_smart_tracker_model(plan_key_prefix, result):
    summary = result.get("summary",{}) or {}
    text = ((summary.get("summary","") or "") + " " + " ".join(summary.get("key_findings",[]) or []) + " " + " ".join(summary.get("recommendations",[]) or [])).lower()
    symptom_level  = st.session_state.get(f"{plan_key_prefix}_symptom_level", 4)
    sleep_hours    = st.session_state.get(f"{plan_key_prefix}_sleep_avg", 6.5)
    medication_done= st.session_state.get(f"{plan_key_prefix}_med_taken", True)
    hydration_ok   = st.session_state.get(f"{plan_key_prefix}_hydration_ok", True)
    food_ok        = st.session_state.get(f"{plan_key_prefix}_food_ok", True)
    movement_ok    = st.session_state.get(f"{plan_key_prefix}_movement_ok", False)
    stress_level   = st.session_state.get(f"{plan_key_prefix}_stress_level", 5)
    risk = 0
    if any(x in text for x in ["nstemi","stemi","cardiac","chest pain","troponin"]): risk += 18
    if any(x in text for x in ["diabetes","glucose","hba1c"]): risk += 10
    if any(x in text for x in ["hypertension","blood pressure"]): risk += 8
    if any(x in text for x in ["urgent","critical","abnormal","elevated"]): risk += 12
    sc = 100 - risk - symptom_level*5 - max(0,7-sleep_hours)*4 - stress_level*2
    sc += 8 if medication_done else -12
    sc += 6 if hydration_ok else -8
    sc += 6 if food_ok else -8
    sc += 5 if movement_ok else 0
    sc = max(5, min(98, int(sc)))
    adh = (25 if medication_done else 0) + (25 if food_ok else 0) + (25 if hydration_ok else 0) + (25 if movement_ok else 0)
    strain = min(100, int(symptom_level*7 + stress_level*5 + max(0,7-sleep_hours)*6))
    if sc >= 75:   status, color = "Ready & stable",         "#22d3a5"
    elif sc >= 50: status, color = "Watch closely",          "#f59e0b"
    else:          status, color = "Needs attention",         "#ef4444"
    alerts = []
    if symptom_level >= 7: alerts.append("Symptoms are high today — slow down and consider medical review.")
    if sleep_hours < 6:    alerts.append("Low sleep may be affecting recovery.")
    if not medication_done: alerts.append("Medicine adherence looks incomplete today.")
    if any(x in text for x in ["chest pain","troponin","nstemi"]) and symptom_level >= 6: alerts.append("Heart-related report — worsening symptoms should not be ignored.")
    if not alerts: alerts.append("Today's check-in looks reasonably stable.")
    weekly = []
    if sc >= 75 and adh >= 75: weekly.append("Next 7 days may be stable with gradual recovery support.")
    if sc < 60:   weekly.append("Next week may be inconsistent unless rest and medicines stay on track.")
    if strain >= 60: weekly.append("High daily strain — monitor activity and symptom load closely.")
    if any(x in text for x in ["diabetes","blood sugar","glucose"]) and not food_ok: weekly.append("Food inconsistency may affect sugar control over the next few days.")
    if not weekly: weekly.append("Maintain current routine and keep follow-up reviews on time.")
    return {"recovery_score":sc,"adherence_score":adh,"strain_score":strain,"status":status,"color":color,"alerts":alerts[:4],"weekly_projection":weekly[:4]}

def build_recovery_forecast(model):
    base, adh, strain = model["recovery_score"], model["adherence_score"], model["strain_score"]
    delta = (2 if adh >= 75 else -2 if adh < 50 else 0) + (1 if strain <= 35 else -2 if strain >= 65 else 0)
    labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    current = base
    scores = []
    for i, day in enumerate(labels):
        current = max(10, min(95, current + delta + ((i % 3)-1)*1.5))
        scores.append({"day":day,"score":int(round(current))})
    return scores

def build_tracker_coach(model, result):
    summary = result.get("summary",{}) or {}
    text = ((summary.get("summary","") or "") + " " + " ".join(summary.get("key_findings",[]) or []) + " " + " ".join(summary.get("recommendations",[]) or [])).lower()
    focus = []
    if model["recovery_score"] < 55: focus.append("Prioritize rest, medicines, hydration, and symptom observation today.")
    if model["strain_score"] > 60:   focus.append("Keep physical activity light — strain is elevated.")
    if model["adherence_score"] < 75: focus.append("Routine consistency is the biggest lever for improvement right now.")
    if any(x in text for x in ["blood sugar","glucose","diabetes"]): focus.append("Meal discipline and sugar monitoring should stay consistent.")
    if any(x in text for x in ["blood pressure","hypertension","cardiac","nstemi","stemi"]): focus.append("Watch for heart and BP warning signs carefully.")
    if not focus: focus.append("Recovery looks stable. Stay consistent with medicines, food, and follow-up.")
    return focus[:4]

def build_plan_risk_flags(result):
    summary = result.get("summary",{}) or {}
    entities = enrich_dashboard_entities(result.get("entities",{}) or {}, summary)
    combined = ((summary.get("summary","") or "") + " " + " ".join(summary.get("key_findings",[]) or []) + " " + " ".join(summary.get("recommendations",[]) or [])).lower()
    flags = []
    if any(x in combined for x in ["chest pain","nstemi","stemi","troponin","cardiac"]): flags.append(("Cardiac caution","Watch for chest discomfort, sweating, or worsening fatigue.","cardiac"))
    if any(x in combined for x in ["diabetes","glucose","blood sugar","hba1c"]): flags.append(("Sugar control","Keep meals regular and monitor sugar if home checks are advised.","metabolic"))
    if any(x in combined for x in ["blood pressure","hypertension","bp"]): flags.append(("BP focus","Salt intake, stress, and medicine routine affect recovery.","pressure"))
    if entities.get("drugs"): flags.append(("Medication review","Keep exact medicine list and timing visible for daily adherence.","medication"))
    if entities.get("lab_values"): flags.append(("Lab follow-up","Important lab signals — may need repeat review or tracking.","lab"))
    if not flags: flags.append(("General recovery","Maintain monitoring, hydration, and doctor follow-up.","general"))
    return flags[:5]

def build_plan_daily_focus(plan, result):
    summary = result.get("summary",{}) or {}
    combined = ((summary.get("summary","") or "") + " " + " ".join(summary.get("key_findings",[]) or []) + " " + " ".join(summary.get("recommendations",[]) or [])).lower()
    focus = []
    if plan["schedule"].get("Morning",{}).get("items"): focus.append("Start the day by confirming medicines and hydration.")
    if any(x in combined for x in ["diabetes","glucose","blood sugar"]): focus.append("Keep meal timing predictable to avoid blood sugar swings.")
    if any(x in combined for x in ["blood pressure","hypertension","cardiac","nstemi","stemi"]): focus.append("Avoid overexertion — keep activity gentle unless doctor cleared progression.")
    if any(x in combined for x in ["infection","fever","weakness"]): focus.append("Recovery depends on rest, hydration, and watching for worsening symptoms.")
    if not focus: focus.append("Stay consistent with medicines, food, movement, hydration, and follow-up.")
    return focus[:4]

def build_plan_adherence_tasks(plan):
    tasks = ["Confirm medicine timing blocks"]
    tasks.append("Track water intake through the day")
    tasks.append("Follow the meal structure suggested below")
    tasks.append("Log symptoms before bedtime")
    tasks.append("Keep follow-up documents ready")
    return tasks[:5]

def build_patient_plan(result):
    summary = result.get("summary",{}) or {}
    entities = enrich_dashboard_entities(result.get("entities",{}) or {}, summary)
    combined = (" ".join((summary.get("key_findings",[]) or []) + (summary.get("recommendations",[]) or [])) + " " + (summary.get("summary","") or "")).lower()
    food = []
    if any(x in combined for x in ["diabetic","glucose","diabetes","blood sugar"]): food.append("Prefer a diabetic-friendly meal plan with controlled sugar.")
    if any(x in combined for x in ["heart","nstemi","stemi","cardiac","blood pressure","hypertension"]): food.append("Follow a low-salt, heart-friendly diet.")
    if any(x in combined for x in ["low-saturated-fat","cholesterol","lipid"]): food.append("Choose low-saturated-fat foods.")
    if not food: food.append("Follow diet advice in the report; confirm with your doctor.")
    activity = []
    if any(x in combined for x in ["no heavy lifting","strenuous activity"]): activity.append("Avoid heavy lifting for the period mentioned in the report.")
    if any(x in combined for x in ["rehabilitation","rehab","cardiac rehabilitation"]): activity.append("Attend prescribed rehabilitation program if doctor approves.")
    if not activity: activity.append("Keep activity gentle unless doctor gave a specific plan.")
    monitoring = []
    if any(x in combined for x in ["blood pressure","hypertension","bp"]): monitoring.append("Track blood pressure regularly if advised.")
    if any(x in combined for x in ["blood sugar","glucose","diabetes","hba1c"]): monitoring.append("Monitor blood sugar regularly if advised.")
    if any(x in combined for x in ["follow-up","review","repeat","2 weeks","7 days"]): monitoring.append("Keep follow-up tests and appointments on time.")
    if not monitoring: monitoring.append("Watch symptoms and attend follow-up as advised.")
    warning_signs = []
    for item in (summary.get("key_findings",[]) or []) + (summary.get("recommendations",[]) or []):
        if any(x in item.lower() for x in ["urgent","seek","chest pain","breathlessness","fainting","sweating","trembling"]): warning_signs.append(item)
    if not warning_signs:
        if any(x in combined for x in ["chest pain","breathlessness","fainting","loss of consciousness"]): warning_signs.append("Seek urgent help if chest pain, breathlessness, fainting, or worsening symptoms occur.")
        else: warning_signs.append("Seek medical help if symptoms suddenly worsen or new serious symptoms appear.")
    return {"schedule":extract_medicine_schedule(result),"diet_plan":build_diet_plan(result),"exercise_plan":build_exercise_plan(result),"food":unique_keep_order(food)[:4],"activity":unique_keep_order(activity)[:4],"monitoring":unique_keep_order(monitoring)[:4],"warning_signs":unique_keep_order(warning_signs)[:4],"doctor_questions":build_treatment_guidance(result)["doctor_questions"][:4],"safety_note":"This plan is educational support only. It does not replace a doctor's prescription."}

def generate_smart_questions(entities, summary_text, doc_type, summary_data=None):
    questions = []
    ents = enrich_dashboard_entities(entities or {}, summary_data or {})
    sd = summary_data or {}
    findings = sd.get("key_findings",[]) or []
    recs     = sd.get("recommendations",[]) or []
    ct = " ".join([summary_text or ""] + findings + recs).lower()
    def add_q(q):
        if q and q not in questions and len(questions) < 6: questions.append(q)
    d  = [x for x in (ents.get("diseases") or [])   if is_useful_question_target(x)]
    dr = [x for x in (ents.get("drugs") or [])       if is_useful_question_target(x)]
    la = [x for x in (ents.get("lab_values") or [])  if is_useful_question_target(x)]
    pr = [x for x in (ents.get("procedures") or [])  if is_useful_question_target(x)]
    sy = [x for x in (ents.get("symptoms") or [])    if is_useful_question_target(x)]
    cf = [x for x in findings                         if is_useful_question_target(x)]
    if cf: add_q(f"Can you explain this finding: {cf[0]}?")
    if d:  add_q(f"What does the report say about {d[0]}?")
    if la: add_q(f"What does the report say about {la[0]}?")
    if pr: add_q(f"What was the result of {pr[0]}?")
    elif any(x in ct for x in ["ecg","echo","ct","mri","x-ray","scan","procedure","test"]): add_q("What tests were performed and what were the important results?")
    if dr: add_q(f"Which medicines are mentioned, such as {dr[0]}?")
    if sy: add_q(f"What symptoms are described, including {sy[0]}?")
    if recs: add_q(f"What follow-up is suggested: {recs[0]}?")
    if any(x in ct for x in ["discharge","treatment plan","plan","rehabilitation","diet"]): add_q("What is the treatment or discharge plan?")
    if any(x in ct for x in ["critical","urgent","emergency","elevated","abnormal","severe"]): add_q("Are there any critical or urgent findings?")
    if any(x in ct for x in ["blood pressure","pulse","heart rate","spo2","physical examination"]): add_q("What vital signs or examination findings are documented?")
    if not questions and summary_text:
        for s in [x.strip() for x in re.split(r"[.!?]+", summary_text) if x.strip()][:3]:
            add_q(f"Can you explain: {s[:120]}?")
    if not questions:
        add_q("What is the most important finding?")
        add_q("What treatment or medication plan is described?")
        add_q("Are there any abnormal findings needing follow-up?")
    return questions[:6]

# ───────────────────────────────────────────────────────────────
# RENDER HELPERS
# ───────────────────────────────────────────────────────────────
def render_structured_report(result):
    summary  = result.get("summary",{})
    entities = result.get("entities",{})
    stats    = result.get("entity_stats",{})
    total_entities = sum(stats.values())
    cr  = summary.get("compression_ratio",0)
    lat = result.get("latency_ms",0)
    orig_len = summary.get("original_length",0)

    # Metric strip
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-box"><div class="metric-num blue">{total_entities}</div><div class="metric-lbl">Medical Details</div></div>
      <div class="metric-box"><div class="metric-num green">{cr:.0%}</div><div class="metric-lbl">Reduction</div></div>
      <div class="metric-box"><div class="metric-num orange">{lat/1000:.1f}s</div><div class="metric-lbl">Processing Time</div></div>
      <div class="metric-box"><div class="metric-num purple">{orig_len}</div><div class="metric-lbl">Words Reviewed</div></div>
    </div>
    """, unsafe_allow_html=True)

    overview_text = summarize_overview(summary, entities)
    st.markdown(f"""
    <div class="card card-blue" style="margin-bottom:0.9rem">
      <div style="font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;margin-bottom:6px">Easy Overview</div>
      <div style="line-height:1.8;color:var(--text);font-size:0.93rem">{overview_text}</div>
    </div>
    """, unsafe_allow_html=True)

    chart_data = {k:len(v) for k,v in entities.items() if v}
    if chart_data:
        ccols = {"diseases":"#fca5a5","drugs":"#93c5fd","symptoms":"#fcd34d","anatomy":"#6ee7cc","procedures":"#c4b5fd","lab_values":"#6ee7d5","other":"#a8bdd6"}
        fig = go.Figure(go.Bar(x=[k.replace("_"," ").title() for k in chart_data], y=list(chart_data.values()),
            marker_color=[ccols.get(k,"#a8bdd6") for k in chart_data],
            text=list(chart_data.values()), textposition="outside", textfont=dict(color="#7a95b8",size=11)))
        fig.update_layout(plot_bgcolor=None, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#7a95b8",family="Inter"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)",showline=False), yaxis=dict(gridcolor="rgba(255,255,255,0.05)",showline=False),
            margin=dict(t=10,b=8,l=8,r=8), height=190, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    tag_map  = {"diseases":"t-disease","drugs":"t-drug","symptoms":"t-symptom","anatomy":"t-anatomy","procedures":"t-procedure","lab_values":"t-lab","other":"t-other"}
    icon_map = {"diseases":"🔴","drugs":"💊","symptoms":"🟡","anatomy":"🫀","procedures":"🔬","lab_values":"🧪","other":"⚪"}
    if any(v for v in entities.values()):
        st.markdown('<div class="sec-title">Entities Found</div>', unsafe_allow_html=True)
        for cat, items in entities.items():
            if items:
                tags_html = "".join([f'<span class="tag {tag_map.get(cat,"t-other")}">{i}</span>' for i in items])
                st.markdown(f"<div style='margin:8px 0'><span style='color:var(--muted);font-size:0.72rem;font-weight:600'>{icon_map.get(cat,'⚪')} {humanize_category(cat)}</span><div style='margin-top:4px'>{tags_html}</div></div>", unsafe_allow_html=True)

    summ_text = summary.get("summary","")
    if summ_text:
        st.markdown('<div class="sec-title">Plain-Language Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""<div class="card" style="padding:1rem"><div style="font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;margin-bottom:7px">Simplified</div><div style="line-height:1.85;font-size:0.92rem">{summ_text}</div></div>""", unsafe_allow_html=True)

    findings = summary.get("key_findings",[])
    if findings:
        st.markdown('<div class="sec-title">What Needs Attention</div>', unsafe_allow_html=True)
        for f in findings:
            st.markdown(f"""<div class="card card-amber" style="padding:10px 14px;margin-bottom:6px"><div style="font-size:0.7rem;color:var(--amber);text-transform:uppercase;letter-spacing:0.08em;font-family:'JetBrains Mono',monospace;margin-bottom:3px">Finding</div><div style="color:var(--text);line-height:1.7;font-size:0.88rem"><span style="color:var(--amber)">▸</span> {f}</div></div>""", unsafe_allow_html=True)

    recs = summary.get("recommendations",[])
    if recs:
        st.markdown('<div class="sec-title">Suggested Next Steps</div>', unsafe_allow_html=True)
        for r in recs:
            st.markdown(f"""<div class="card card-cyan" style="padding:10px 14px;margin-bottom:6px"><div style="font-size:0.7rem;color:var(--cyan);text-transform:uppercase;letter-spacing:0.08em;font-family:'JetBrains Mono',monospace;margin-bottom:3px">Next Step</div><div style="color:var(--text);line-height:1.7;font-size:0.88rem"><span style="color:var(--cyan)">✓</span> {r}</div></div>""", unsafe_allow_html=True)


def render_patient_care_dashboard(result, doc_id, doc_type):
    summary  = result.get("summary",{}) or {}
    entities = enrich_dashboard_entities(result.get("entities",{}) or {}, summary)
    findings = summary.get("key_findings",[]) or []
    recs     = summary.get("recommendations",[]) or []
    attention        = build_attention_profile(result)
    story            = build_report_story(result)
    treatment_guidance = build_treatment_guidance(result)
    total_details    = get_total_medical_details(entities)
    report_mode      = infer_report_mode(result, doc_type)

    if total_details == 0:
        st.markdown(f"""<div class="card card-amber" style="padding:1.2rem"><div style="font-size:0.95rem;font-weight:700;margin-bottom:8px">⚠ No strong medical structure detected</div><div style="font-size:0.88rem;line-height:1.8;color:var(--muted)">File <b style="color:var(--text)">{doc_id or "uploaded document"}</b> did not yield enough medical entities. Try a discharge summary, lab report, prescription, or scan report.</div></div>""", unsafe_allow_html=True)
        return

    # ── HERO REPORT CARD ──────────────────────────────────────────
    attn_color_map = {"High attention":"var(--red)","Moderate attention":"var(--amber)","Routine attention":"var(--green)"}
    ac = attn_color_map.get(attention["level"],"var(--cyan)")
    h_col, a_col = st.columns([1.5,1], gap="large")
    with h_col:
        entity_chips = "".join([f'<span style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.2);border-radius:999px;padding:3px 10px;font-size:0.72rem;color:#93c5fd;margin-right:5px">{humanize_category(k)}: {len(v)}</span>' for k,v in entities.items() if v and k in ["diseases","drugs","symptoms","lab_values","procedures"]])
        st.markdown(f"""
        <div class="card card-blue" style="padding:1.2rem 1.4rem">
          <div style="font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.12em;font-family:'JetBrains Mono',monospace;margin-bottom:6px">Active Report</div>
          <div style="font-size:1.25rem;font-weight:800;color:var(--text);letter-spacing:-0.02em">{doc_id or "Medical Report"}</div>
          <div style="font-size:0.82rem;color:var(--muted);margin-top:4px">Type: <span style="color:var(--blue)">{report_mode}</span></div>
          <div style="margin-top:10px">{entity_chips}</div>
        </div>
        """, unsafe_allow_html=True)
    with a_col:
        st.markdown(f"""
        <div class="card" style="padding:1.2rem 1.4rem;border-color:{ac.replace('var(','').replace(')','')};border-color:{ac}">
          <div style="font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;margin-bottom:8px">Attention Level</div>
          <div style="font-size:1.05rem;font-weight:700;color:{ac}">{attention["level"]}</div>
          <div style="font-size:0.83rem;color:var(--muted);line-height:1.6;margin-top:8px">{attention["note"]}</div>
          {"".join([f'<div style="margin-top:5px;font-size:0.78rem;color:var(--muted);padding:4px 0;border-top:1px solid var(--stroke)">▸ {a}</div>' for a in attention["focus_areas"][:2]])}
        </div>
        """, unsafe_allow_html=True)

    # ── METRIC STRIP ──────────────────────────────────────────────
    metric_items = [("Conditions",len(entities.get("diseases",[])),"var(--red)"),("Medicines",len(entities.get("drugs",[])),"var(--blue)"),("Symptoms",len(entities.get("symptoms",[])),"var(--amber)"),("Lab Signals",len(entities.get("lab_values",[])),"var(--cyan)")]
    st.markdown("<div class='metric-grid'>" + "".join([f"<div class='metric-box'><div class='metric-num' style='color:{c}'>{v}</div><div class='metric-lbl'>{l}</div></div>" for l,v,c in metric_items]) + "</div>", unsafe_allow_html=True)

    # ── FLOW STEPS ────────────────────────────────────────────────
    st.markdown('<div class="sec-title">Report Processing Flow</div>', unsafe_allow_html=True)
    flow = [("01","Report Ingested",f"Linked to {doc_id or 'this case'}"),("02","Details Extracted",f"{total_details} structured medical details found"),("03","Summary Created",f"{len(findings)} key findings captured"),("04","Follow-up Built",f"{len(recs)} recommendations generated")]
    fcols = [("#3b82f6","var(--blue)"),("#8b5cf6","var(--violet)"),("#f59e0b","var(--amber)"),("#22d3a5","var(--green)")]
    fc = st.columns(4, gap="small")
    for col, (num,title,desc),(hc,tc) in zip(fc,flow,fcols):
        with col:
            st.markdown(f"""<div style="background:var(--panel);border:1px solid var(--stroke);border-top:2px solid {hc};border-radius:14px;padding:14px;min-height:100px"><div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:800;color:{hc};opacity:0.3;line-height:1">{num}</div><div style="font-size:0.82rem;font-weight:700;color:var(--text);margin:5px 0 3px">{title}</div><div style="font-size:0.77rem;color:var(--muted);line-height:1.5">{desc}</div></div>""", unsafe_allow_html=True)

    # ── HIGHLIGHTS + CARE OUTLOOK ─────────────────────────────────
    lc, rc = st.columns(2, gap="large")
    with lc:
        st.markdown('<div class="sec-title">Report Highlights</div>', unsafe_allow_html=True)
        for item in story:
            st.markdown(f"""<div style="background:var(--panel);border:1px solid var(--stroke);border-left:2px solid var(--blue);border-radius:10px;padding:10px 14px;margin-bottom:6px"><div style="font-size:0.86rem;color:var(--text);line-height:1.7">{item}</div></div>""", unsafe_allow_html=True)
        if not story:
            st.markdown('<div class="card" style="padding:0.9rem"><div style="color:var(--muted);font-size:0.85rem">Limited highlights extracted.</div></div>', unsafe_allow_html=True)
    with rc:
        st.markdown('<div class="sec-title">Care Outlook</div>', unsafe_allow_html=True)
        for item in attention["focus_areas"]:
            st.markdown(f"""<div style="background:var(--panel);border:1px solid var(--stroke);border-left:2px solid var(--violet);border-radius:10px;padding:10px 14px;margin-bottom:6px"><div style="font-size:0.86rem;color:var(--text);line-height:1.7">{item}</div></div>""", unsafe_allow_html=True)

    # ── ENTITY CHART ──────────────────────────────────────────────
    chart_data = {k:len(v) for k,v in entities.items() if v and k in {"diseases","drugs","symptoms","lab_values","procedures","anatomy"}}
    if chart_data:
        ccols2 = {"diseases":"#fca5a5","drugs":"#93c5fd","symptoms":"#fcd34d","anatomy":"#6ee7cc","procedures":"#c4b5fd","lab_values":"#6ee7d5"}
        fig = go.Figure(go.Bar(x=[humanize_category(k) for k in chart_data], y=list(chart_data.values()),
            marker_color=[ccols2.get(k,"#a8bdd6") for k in chart_data],
            text=list(chart_data.values()), textposition="outside", textfont=dict(color="#7a95b8",size=11)))
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#7a95b8",family="Inter"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.04)",showline=False), yaxis=dict(gridcolor="rgba(255,255,255,0.04)",showline=False),
            margin=dict(t=16,b=8,l=8,r=8), height=230, showlegend=False,
            title=dict(text="Medical Details by Category",font=dict(color="#7a95b8",size=12)))
        st.plotly_chart(fig, use_container_width=True)

    # ── DETAILED MENTIONS + NEXT STEPS ───────────────────────────
    dc, nc = st.columns([1.2,1], gap="large")
    with dc:
        st.markdown('<div class="sec-title">Detailed Mentions</div>', unsafe_allow_html=True)
        tag_map2 = {"diseases":"t-disease","drugs":"t-drug","symptoms":"t-symptom","lab_values":"t-lab","procedures":"t-procedure","anatomy":"t-anatomy"}
        any_d = False
        for cat in ["diseases","symptoms","drugs","lab_values","procedures","anatomy","other"]:
            items = entities.get(cat) or []
            if items:
                any_d = True
                tags = "".join([f"<span class='tag {tag_map2.get(cat,'t-other')}'>{i}</span>" for i in items[:12]])
                st.markdown(f"""<div class="card" style="padding:10px 14px;margin-bottom:7px"><div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;font-family:'JetBrains Mono',monospace;margin-bottom:7px">{humanize_category(cat)}</div><div style="line-height:2">{tags}</div></div>""", unsafe_allow_html=True)
        if not any_d: st.info("No structured medical details extracted yet.")
    with nc:
        st.markdown('<div class="sec-title">Findings & Next Steps</div>', unsafe_allow_html=True)
        for item in findings[:5]:
            st.markdown(f"""<div class="card card-amber" style="padding:10px 14px;margin-bottom:6px"><div style="font-size:0.68rem;color:var(--amber);text-transform:uppercase;letter-spacing:0.08em;font-family:'JetBrains Mono',monospace;margin-bottom:3px">Finding</div><div style="font-size:0.83rem;color:var(--text);line-height:1.6">{item}</div></div>""", unsafe_allow_html=True)
        for item in recs[:5]:
            st.markdown(f"""<div class="card card-cyan" style="padding:10px 14px;margin-bottom:6px"><div style="font-size:0.68rem;color:var(--cyan);text-transform:uppercase;letter-spacing:0.08em;font-family:'JetBrains Mono',monospace;margin-bottom:3px">Next Step</div><div style="font-size:0.83rem;color:var(--text);line-height:1.6">{item}</div></div>""", unsafe_allow_html=True)
        if not findings and not recs: st.info("No explicit next-step guidance extracted.")

    # ── TREATMENT ─────────────────────────────────────────────────
    st.markdown('<div class="sec-title">Treatment Discussion Support</div>', unsafe_allow_html=True)
    st.markdown(f"""<div class="card card-red" style="padding:10px 16px;margin-bottom:10px;font-size:0.82rem;color:#fca5a5;line-height:1.6">⚠️ {treatment_guidance["warning"]}</div>""", unsafe_allow_html=True)
    t1,t2,t3 = st.columns(3, gap="large")
    with t1:
        st.markdown('<div class="sec-title">What to Discuss</div>', unsafe_allow_html=True)
        for item in treatment_guidance["treatment_topics"]:
            st.markdown(f"""<div class="card" style="padding:10px 14px;margin-bottom:6px;font-size:0.83rem;color:var(--muted);line-height:1.6">{item}</div>""", unsafe_allow_html=True)
    with t2:
        st.markdown('<div class="sec-title">Ask Your Doctor</div>', unsafe_allow_html=True)
        for item in treatment_guidance["doctor_questions"]:
            st.markdown(f"""<div class="card card-amber" style="padding:10px 14px;margin-bottom:6px;font-size:0.83rem;color:#fcd34d;line-height:1.6">❓ {item}</div>""", unsafe_allow_html=True)
    with t3:
        st.markdown('<div class="sec-title">Lifestyle Support</div>', unsafe_allow_html=True)
        for item in treatment_guidance["lifestyle_support"]:
            st.markdown(f"""<div class="card card-cyan" style="padding:10px 14px;margin-bottom:6px;font-size:0.83rem;color:#6ee7cc;line-height:1.6">✓ {item}</div>""", unsafe_allow_html=True)


def render_patient_plan(result):
    plan              = build_patient_plan(result)
    risk_flags        = build_plan_risk_flags(result)
    plan_focus        = build_plan_daily_focus(plan, result)
    adherence_tasks   = build_plan_adherence_tasks(plan)
    plan_key_prefix   = f"patient_plan_{st.session_state.get('doc_id','default')}"
    entities          = enrich_dashboard_entities(result.get("entities",{}) or {}, result.get("summary",{}) or {})
    diseases          = entities.get("diseases",[]) or []
    drugs             = entities.get("drugs",[]) or []
    total_meds        = sum(len(v.get("items",[])) for v in plan["schedule"].values())
    total_monitoring  = len(plan["monitoring"])
    total_warnings    = len(plan["warning_signs"])
    total_exercises   = len(plan["exercise_plan"])

    # ── WELCOME BANNER ────────────────────────────────────────────
    diag_str = diseases[0] if diseases else "your report"
    st.markdown(f"""
    <div class="welcome-banner">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:12px">
        <div>
          <div style="font-size:0.72rem;color:#6ee7cc;text-transform:uppercase;letter-spacing:0.12em;font-family:'JetBrains Mono',monospace;margin-bottom:6px">● Recovery Active</div>
          <div class="welcome-title">Your Daily <span style="color:#29d9c2">Recovery Plan</span></div>
          <div class="welcome-sub">Personalised guidance from your medical report — routines, medicines, diet, exercises and smart tracking all in one place.</div>
          <div style="margin-top:12px">{"".join([f'<span class="plan-pill">{t}</span>' for t in adherence_tasks[:4]])}</div>
        </div>
        <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);border-radius:14px;padding:12px 16px;max-width:300px;font-size:0.79rem;color:#fca5a5;line-height:1.6">
          ⚠️ {plan['safety_note']}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI STRIP ─────────────────────────────────────────────────
    kpi_items = [("💊",total_meds,"Medicine Events","rgba(59,130,246,0.12)","var(--blue)"),("📊",total_monitoring,"Monitoring Actions","rgba(34,211,165,0.12)","var(--green)"),("🏃",total_exercises,"Exercise Tracks","rgba(139,92,246,0.12)","var(--violet)"),("🚨",total_warnings,"Warning Signals","rgba(239,68,68,0.12)","var(--red)")]
    k_cols = st.columns(4, gap="small")
    for col, (icon,val,lbl,bg,color) in zip(k_cols, kpi_items):
        with col:
            st.markdown(f"""<div class="stat-pill"><div class="stat-icon" style="background:{bg}">{icon}</div><div><div class="stat-val" style="color:{color}">{val}</div><div class="stat-lbl">{lbl}</div></div></div>""", unsafe_allow_html=True)

    # ── SMART TRACKER (top section) ───────────────────────────────
    st.markdown('<div class="sec-title">Smart Recovery Tracker</div>', unsafe_allow_html=True)
    check_col, score_col = st.columns([1,1.3], gap="large")
    with check_col:
        c1, c2 = st.columns(2)
        with c1:
            st.checkbox("Medicines taken", value=True, key=f"{plan_key_prefix}_med_taken")
            st.checkbox("Diet followed", value=True, key=f"{plan_key_prefix}_food_ok")
        with c2:
            st.checkbox("Hydration goal", value=True, key=f"{plan_key_prefix}_hydration_ok")
            st.checkbox("Exercise done", value=False, key=f"{plan_key_prefix}_movement_ok")
        st.slider("Symptom level (0=none)", 0, 10, 4, key=f"{plan_key_prefix}_symptom_level")
        st.slider("Stress level", 0, 10, 5, key=f"{plan_key_prefix}_stress_level")
        st.slider("Sleep hours", 0.0, 12.0, 6.5, step=0.5, key=f"{plan_key_prefix}_sleep_avg")

    smart    = build_smart_tracker_model(plan_key_prefix, result)
    forecast = build_recovery_forecast(smart)
    coach    = build_tracker_coach(smart, result)
    sc       = smart["recovery_score"]
    scolor   = smart["color"]

    with score_col:
        # Gauge chart
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sc,
            title={"text":"Recovery Score","font":{"color":"#7a95b8","size":12}},
            number={"font":{"color":scolor,"size":40,"family":"Inter"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#364a62","nticks":5},
                "bar":{"color":scolor,"thickness":0.22},
                "bgcolor":"rgba(0,0,0,0)",
                "borderwidth":0,
                "steps":[{"range":[0,40],"color":"rgba(239,68,68,0.12)"},{"range":[40,70],"color":"rgba(245,158,11,0.1)"},{"range":[70,100],"color":"rgba(34,211,165,0.1)"}],
                "threshold":{"line":{"color":scolor,"width":3},"thickness":0.75,"value":sc}
            }
        ))
        gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7a95b8",family="Inter"), height=220,
            margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(gauge, use_container_width=True)
        # Mini metrics
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-top:4px">
          <div class="card" style="padding:10px;text-align:center"><div style="font-size:1.2rem;font-weight:800;color:var(--blue);font-family:'JetBrains Mono',monospace">{smart['adherence_score']}</div><div style="font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em">Adherence</div></div>
          <div class="card" style="padding:10px;text-align:center"><div style="font-size:1.2rem;font-weight:800;color:var(--amber);font-family:'JetBrains Mono',monospace">{smart['strain_score']}</div><div style="font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em">Strain</div></div>
          <div class="card" style="padding:10px;text-align:center"><div style="font-size:0.88rem;font-weight:700;color:{scolor};line-height:1.2">{smart['status']}</div><div style="font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em">Status</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── 7-DAY FORECAST CHART ──────────────────────────────────────
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=[p["day"] for p in forecast], y=[p["score"] for p in forecast],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2.5, shape="spline"),
        marker=dict(size=7, color="#22d3a5", line=dict(color="#3b82f6",width=1.5)),
        fill="tozeroy", fillcolor="rgba(59,130,246,0.07)", name="Recovery"))
    trend_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7a95b8",family="Inter"),
        height=200, margin=dict(t=10,b=20,l=30,r=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)",showline=False),
        yaxis=dict(range=[0,100],gridcolor="rgba(255,255,255,0.04)",showline=False,title="Score"),
        showlegend=False)
    st.markdown('<div class="sec-title">7-Day Recovery Forecast</div>', unsafe_allow_html=True)
    st.plotly_chart(trend_fig, use_container_width=True)

    # ── WATCHLIST + DAILY FOCUS ───────────────────────────────────
    wl_col, df_col = st.columns([1,1], gap="large")
    with wl_col:
        st.markdown('<div class="sec-title">Priority Watchlist</div>', unsafe_allow_html=True)
        for idx, flag in enumerate(risk_flags, 1):
            st.markdown(f"""<div class="plan-step"><div class="plan-step-num">{idx}</div><div class="plan-step-text"><div style="font-weight:700;color:var(--text);margin-bottom:3px">{flag[0]}</div><div style="color:var(--muted)">{flag[1]}</div></div></div>""", unsafe_allow_html=True)
    with df_col:
        st.markdown('<div class="sec-title">Today\'s Recovery Focus</div>', unsafe_allow_html=True)
        for item in plan_focus:
            st.markdown(f"""<div class="plan-item-card"><div class="plan-item-text">{item}</div></div>""", unsafe_allow_html=True)
        st.markdown('<div class="sec-title" style="margin-top:0.8rem">Smart Coach</div>', unsafe_allow_html=True)
        for item in coach:
            st.markdown(f"""<div class="plan-item-card"><div class="plan-item-text">{item}</div></div>""", unsafe_allow_html=True)

    # ── DAILY SCHEDULE ────────────────────────────────────────────
    st.markdown('<div class="sec-title">Daily Medicine & Routine Schedule</div>', unsafe_allow_html=True)
    slot_colors = {"Morning":"#f59e0b","Afternoon":"#3b82f6","Evening":"#8b5cf6","Night":"#22d3a5"}
    slot_times  = {"Morning":"06:00–09:00","Afternoon":"12:00–15:00","Evening":"17:00–19:00","Night":"20:00–22:00"}
    t_cols = st.columns(4, gap="small")
    for col, slot in zip(t_cols, ["Morning","Afternoon","Evening","Night"]):
        payload = plan["schedule"].get(slot,{"time":"","items":[]})
        items   = payload.get("items",[])
        color   = slot_colors[slot]
        with col:
            body = "".join([f"<div class='timeline-entry'><div class='timeline-dot' style='background:rgba({','.join(str(int(color.lstrip('#')[i:i+2],16)) for i in (0,2,4))},0.15)'>•</div><div><div class='timeline-time'>{payload['time']}</div><div class='timeline-text'>{item}</div></div></div>" for item in items[:4]]) if items else f"<div class='plan-item-text' style='color:var(--muted)'>No schedule extracted.</div>"
            st.markdown(f"""<div class="card" style="padding:0.9rem;min-height:200px;border-top:2px solid {color}"><div style="font-size:0.78rem;font-weight:700;color:{color};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;font-family:'JetBrains Mono',monospace">{slot} <span style="color:var(--muted);font-weight:400">{slot_times[slot]}</span></div>{body}</div>""", unsafe_allow_html=True)

    # ── HABIT TRACKER STRIP ───────────────────────────────────────
    st.markdown('<div class="sec-title">Habit & Adherence Tracker</div>', unsafe_allow_html=True)
    adherence_pct = smart["adherence_score"]
    sleep_h       = st.session_state.get(f"{plan_key_prefix}_sleep_avg", 6.5)
    water_pct     = 100 if st.session_state.get(f"{plan_key_prefix}_hydration_ok", True) else 40
    med_pct       = 100 if st.session_state.get(f"{plan_key_prefix}_med_taken", True) else 0
    move_pct      = 100 if st.session_state.get(f"{plan_key_prefix}_movement_ok", False) else 15
    habit_items = [
        ("💊",f"{med_pct}%","Medicine","rgba(59,130,246,0.15)","#3b82f6",med_pct),
        ("🥗",f"{adherence_pct}%","Diet","rgba(34,211,165,0.15)","#22d3a5",adherence_pct),
        ("💧",f"{water_pct}%","Hydration","rgba(41,217,194,0.15)","#29d9c2",water_pct),
        ("🏃",f"{move_pct}%","Movement","rgba(139,92,246,0.15)","#8b5cf6",move_pct),
        ("😴",f"{sleep_h}h","Sleep","rgba(245,158,11,0.15)","#f59e0b",int(sleep_h/8*100)),
    ]
    h_cols = st.columns(5, gap="small")
    for col, (icon, val, lbl, bg, color, pct) in zip(h_cols, habit_items):
        with col:
            st.markdown(f"""
            <div class="habit-card">
              <div class="habit-icon" style="background:{bg}">{icon}</div>
              <div class="habit-val">{val}</div>
              <div class="habit-lbl">{lbl}</div>
              <div class="prog-wrap"><div class="prog-fill" style="width:{min(100,pct)}%;background:{color}"></div></div>
            </div>
            """, unsafe_allow_html=True)

    # ── WEEKLY WELLNESS CHART ─────────────────────────────────────
    st.markdown('<div class="sec-title">Weekly Wellness Tracker</div>', unsafe_allow_html=True)
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    rec_scores = [forecast[i]["score"] for i in range(7)]
    symptom_scores = [st.session_state.get(f"{plan_key_prefix}_symptom_level",4)] * 7
    w_fig = go.Figure()
    w_fig.add_trace(go.Scatter(x=days, y=rec_scores, mode="lines+markers", name="Recovery",
        line=dict(color="#29d9c2",width=2.5,shape="spline"),
        marker=dict(size=6,color="#29d9c2"),
        fill="tozeroy", fillcolor="rgba(41,217,194,0.06)"))
    w_fig.add_trace(go.Scatter(x=days, y=[s*10 for s in symptom_scores], mode="lines", name="Symptom×10",
        line=dict(color="#f59e0b",width=1.5,dash="dot")))
    w_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#7a95b8",family="Inter"), height=200,
        margin=dict(t=10,b=20,l=30,r=10),
        xaxis=dict(gridcolor="rgba(255,255,255,0.04)",showline=False),
        yaxis=dict(range=[0,100],gridcolor="rgba(255,255,255,0.04)",showline=False),
        legend=dict(font=dict(color="#7a95b8",size=10),bgcolor="rgba(0,0,0,0)",orientation="h",y=-0.3))
    st.plotly_chart(w_fig, use_container_width=True)

    # ── DIET PLANNER ──────────────────────────────────────────────
    st.markdown('<div class="sec-title">Precision Diet Planner</div>', unsafe_allow_html=True)
    meal_colors = {"Breakfast":"#f59e0b","Lunch":"#3b82f6","Healthy Snacks":"#8b5cf6","Dinner":"#22d3a5","Drinks":"#29d9c2"}
    meal_icons  = {"Breakfast":"☀","Lunch":"🌤","Healthy Snacks":"🍎","Dinner":"🌙","Drinks":"💧"}
    d_cols = st.columns(5, gap="small")
    for col, meal_name in zip(d_cols, ["Breakfast","Lunch","Healthy Snacks","Dinner","Drinks"]):
        items = plan["diet_plan"].get(meal_name,[])
        color = meal_colors[meal_name]
        with col:
            items_html = "".join([f"<li>{item}</li>" for item in items]) or "<li style='color:var(--muted)'>No suggestion.</li>"
            st.markdown(f"""<div class="card" style="padding:0.85rem;min-height:195px;border-top:2px solid {color}"><div style="font-size:0.72rem;font-weight:700;color:{color};text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;font-family:'JetBrains Mono',monospace">{meal_icons[meal_name]} {meal_name}</div><ul class="plan-list">{items_html}</ul></div>""", unsafe_allow_html=True)

    # ── HEALTH REPORTS STRIP (from Image 1) ──────────────────────
    st.markdown('<div class="sec-title">Health Reports & Progress</div>', unsafe_allow_html=True)
    hr_c1, hr_c2, hr_c3 = st.columns([1,1,1], gap="large")
    with hr_c1:
        # Donut gauge: Monthly progress
        mp = smart["recovery_score"]
        mp_color = "#22d3a5" if mp >= 70 else "#f59e0b" if mp >= 50 else "#ef4444"
        donut = go.Figure(go.Pie(values=[mp, 100-mp], hole=0.7, showlegend=False,
            marker_colors=[mp_color,"rgba(255,255,255,0.06)"],
            textinfo="none"))
        donut.add_annotation(text=f"<b>{mp}%</b>", x=0.5, y=0.5, font=dict(color=mp_color,size=22,family="Inter"), showarrow=False)
        donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=8,b=8,l=8,r=8), height=140)
        st.plotly_chart(donut, use_container_width=True)
        st.markdown(f"<div style='text-align:center;font-size:0.8rem;color:var(--muted);margin-top:-8px'>You've reached <b style='color:{mp_color}'>{mp}%</b> of your therapeutic target</div>", unsafe_allow_html=True)
    with hr_c2:
        # Sleep quality donut (from Image 1)
        sleep_h2 = st.session_state.get(f"{plan_key_prefix}_sleep_avg", 6.5)
        sleep_pct = int(sleep_h2/8*100)
        s_color = "#22d3a5" if sleep_pct >= 80 else "#f59e0b" if sleep_pct >= 60 else "#ef4444"
        sleep_donut = go.Figure(go.Pie(values=[sleep_pct,100-sleep_pct], hole=0.65, showlegend=False,
            marker_colors=[s_color,"rgba(255,255,255,0.06)"], textinfo="none"))
        sleep_donut.add_annotation(text=f"<b>{sleep_h2}h</b>", x=0.5, y=0.55, font=dict(color=s_color,size=18,family="Inter"), showarrow=False)
        sleep_donut.add_annotation(text="Sleep", x=0.5, y=0.35, font=dict(color="#7a95b8",size=10,family="Inter"), showarrow=False)
        sleep_donut.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=8,b=8,l=8,r=8), height=140)
        st.plotly_chart(sleep_donut, use_container_width=True)
        st.markdown(f"<div style='text-align:center;font-size:0.8rem;color:var(--muted);margin-top:-8px'>Sleep quality &nbsp;·&nbsp; <b style='color:{s_color}'>{sleep_pct}%</b> of target</div>", unsafe_allow_html=True)
    with hr_c3:
        # Report highlights list
        st.markdown('<div style="margin-top:4px"></div>', unsafe_allow_html=True)
        report_items = [
            ("Weight change", "78% monitored", "#22d3a5"),
            ("General health", "Stable trend", "#3b82f6"),
            ("Medicine adherence", f"{smart['adherence_score']}% on track", "#8b5cf6"),
        ]
        for label, val, color in report_items:
            pct = int(val.split("%")[0]) if "%" in val else 75
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:8px 0;border-bottom:1px solid var(--stroke)">
              <svg width="32" height="32" viewBox="0 0 32 32"><circle cx="16" cy="16" r="13" fill="none" stroke="rgba(255,255,255,0.08)" stroke-width="3"/><circle cx="16" cy="16" r="13" fill="none" stroke="{color}" stroke-width="3" stroke-dasharray="{int(82*pct/100)} 82" stroke-linecap="round" transform="rotate(-90 16 16)"/></svg>
              <div><div style="font-size:0.83rem;font-weight:600;color:var(--text)">{label}</div><div style="font-size:0.72rem;color:{color}">{val}</div></div>
            </div>
            """, unsafe_allow_html=True)

    # ── EXERCISE PLANS ────────────────────────────────────────────
    st.markdown('<div class="sec-title">Guided Exercise Routine</div>', unsafe_allow_html=True)
    ex_cols = st.columns(len(plan["exercise_plan"]), gap="large")
    for col, routine in zip(ex_cols, plan["exercise_plan"]):
        with col:
            steps_html = "".join([f"<div class='plan-step'><div class='plan-step-num'>{idx}</div><div class='plan-step-text'>{step}</div></div>" for idx,step in enumerate(routine["steps"],1)])
            st.markdown(f"""<div class="card card-violet" style="padding:1rem;min-height:240px"><div class="plan-section-heading">{routine['title']}</div>{steps_html}</div>""", unsafe_allow_html=True)

    # ── MONITORING + WARNINGS ─────────────────────────────────────
    mon_col, warn_col = st.columns(2, gap="large")
    with mon_col:
        st.markdown('<div class="sec-title">Monitoring Plan</div>', unsafe_allow_html=True)
        for item in plan["monitoring"]:
            st.markdown(f"""<div class="plan-item-card" style="border-left:2px solid var(--blue);"><div class="plan-item-text">📊 {item}</div></div>""", unsafe_allow_html=True)
    with warn_col:
        st.markdown('<div class="sec-title">⚠ Warning Signs</div>', unsafe_allow_html=True)
        for item in plan["warning_signs"]:
            st.markdown(f"""<div class="plan-item-card" style="border-left:2px solid var(--red);background:rgba(239,68,68,0.05)"><div class="plan-item-text" style="color:#fca5a5">🚨 {item}</div></div>""", unsafe_allow_html=True)

    # ── APPOINTMENTS + OUTLOOK ────────────────────────────────────
    appt_col, out_col = st.columns([1,1.1], gap="large")
    with appt_col:
        st.markdown('<div class="sec-title">Reminders & Follow-ups</div>', unsafe_allow_html=True)
        appts = [
            ("Follow-up Consultation","Review recovery progress","👨‍⚕️ Attending Doctor","#3b82f6"),
            ("Blood Test / Lab Work","Monitor report lab values","🧪 Pathology Lab","#f59e0b"),
            ("Medicine Review","Check dosing schedule","💊 Doctor / Pharmacist","#8b5cf6"),
        ]
        if any(x in " ".join(diseases).lower() for x in ["cardiac","heart","nstemi","stemi"]):
            appts.insert(0,("Cardiac Review","ECG / Echo follow-up","❤️ Cardiologist","#ef4444"))
        for title, desc, doctor, color in appts[:3]:
            st.markdown(f"""
            <div style="background:var(--panel);border:1px solid var(--stroke);border-radius:14px;padding:12px 14px;margin-bottom:8px;display:flex;gap:12px;align-items:center">
              <div style="width:36px;height:36px;background:rgba(59,130,246,0.1);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:15px;flex-shrink:0">📋</div>
              <div style="flex:1">
                <div style="font-size:0.84rem;font-weight:700;color:var(--text)">{title}</div>
                <div style="font-size:0.73rem;color:var(--muted);margin-top:1px">{desc}</div>
                <div style="font-size:0.71rem;color:{color};margin-top:3px">{doctor}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
    with out_col:
        st.markdown('<div class="sec-title">Weekly Recovery Outlook</div>', unsafe_allow_html=True)
        for week, label, pct in [("Week 1","Stabilisation — rest + medicines",60),("Week 2","Light activity, walks",72),("Week 3","Strength recovery",82),("Week 4","Follow-up review",90)]:
            bc = "#22d3a5" if pct>=80 else "#3b82f6" if pct>=65 else "#f59e0b"
            st.markdown(f"""
            <div style="padding:10px 0;border-bottom:1px solid var(--stroke)">
              <div style="display:flex;justify-content:space-between;margin-bottom:5px">
                <div><span style="font-size:0.68rem;color:var(--muted);font-family:'JetBrains Mono',monospace;text-transform:uppercase">{week}</span>
                     <span style="font-size:0.78rem;color:var(--muted);margin-left:8px">{label}</span></div>
                <span style="font-size:0.73rem;color:{bc};font-family:'JetBrains Mono',monospace;font-weight:700">{pct}%</span>
              </div>
              <div class="prog-wrap"><div class="prog-fill" style="width:{pct}%;background:{bc}"></div></div>
            </div>
            """, unsafe_allow_html=True)

    # ── SMART ALERTS + DOCTOR Q ───────────────────────────────────
    al_col, dq_col = st.columns(2, gap="large")
    alert_styles = [("rgba(59,130,246,0.08)","rgba(59,130,246,0.22)","#93c5fd"),("rgba(245,158,11,0.08)","rgba(245,158,11,0.22)","#fcd34d"),("rgba(239,68,68,0.08)","rgba(239,68,68,0.22)","#fca5a5"),("rgba(34,211,165,0.08)","rgba(34,211,165,0.22)","#6ee7cc")]
    with al_col:
        st.markdown('<div class="sec-title">Smart Alerts</div>', unsafe_allow_html=True)
        for item, (bg,border,color) in zip(smart["alerts"], alert_styles):
            st.markdown(f"""<div style="background:{bg};border:1px solid {border};border-radius:10px;padding:10px 14px;margin-bottom:6px;font-size:0.82rem;color:{color};line-height:1.5">{item}</div>""", unsafe_allow_html=True)
    with dq_col:
        st.markdown('<div class="sec-title">Ask Your Doctor</div>', unsafe_allow_html=True)
        for item in plan["doctor_questions"]:
            st.markdown(f"""<div style="display:flex;gap:8px;padding:10px 12px;background:var(--panel);border:1px solid var(--stroke);border-radius:10px;margin-bottom:6px"><span style="color:var(--violet);flex-shrink:0">❓</span><span style="font-size:0.82rem;color:var(--muted);line-height:1.5">{item}</span></div>""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# SIDEBAR
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    api_ok = check_api()

    # Logo
    st.markdown("""
    <div style="padding:4px 4px 18px;border-bottom:1px solid rgba(59,130,246,0.15);margin-bottom:18px">
      <div style="display:flex;align-items:center;gap:10px">
        <div style="width:38px;height:38px;background:linear-gradient(135deg,#3b82f6,#06b6d4);border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:18px;flex-shrink:0">🏥</div>
        <div>
          <div style="font-size:1rem;font-weight:800;color:#e8f1ff">MedAnalyzer</div>
          <div style="font-size:0.68rem;color:#7a95b8;letter-spacing:0.06em">AI Health Platform</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # API status
    dot = "dot-green" if api_ok else "dot-red"
    st.markdown(f"""
    <div style="background:{'rgba(34,211,165,0.07)' if api_ok else 'rgba(239,68,68,0.07)'};border:1px solid {'rgba(34,211,165,0.2)' if api_ok else 'rgba(239,68,68,0.2)'};border-radius:10px;padding:9px 12px;margin-bottom:16px;display:flex;align-items:center;gap:8px">
      <span class="status-dot {dot}"></span>
      <div>
        <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;font-family:'JetBrains Mono',monospace">API Status</div>
        <div style="font-size:0.83rem;font-weight:700;color:{'#22d3a5' if api_ok else '#ef4444'}">{'Connected' if api_ok else 'Offline'}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Document type
    st.markdown('<div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;font-family:\'JetBrains Mono\',monospace">Document Type</div>', unsafe_allow_html=True)
    doc_type = st.selectbox("Document Type", ["general","lab_report","prescription","discharge_summary","radiology","pathology"], label_visibility="collapsed")

    # Nav links
    st.markdown('<div style="margin:18px 0 10px;border-top:1px solid var(--stroke)"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;font-family:\'JetBrains Mono\',monospace">Platform Stack</div>', unsafe_allow_html=True)

    for icon, name, desc in [("🤗","HuggingFace","NER + Summarization"),("🔗","LangChain","RAG Pipeline"),("🗄️","ChromaDB","Vector Store"),("⚡","FastAPI","Backend"),("📈","MLflow","MLOps Tracking")]:
        st.markdown(f"""<div style="display:flex;align-items:center;gap:8px;padding:7px 8px;border-radius:9px;margin-bottom:3px;background:rgba(255,255,255,0.02)"><span style="font-size:14px">{icon}</span><div><div style="font-size:0.79rem;font-weight:600;color:#c8dcff">{name}</div><div style="font-size:0.68rem;color:var(--muted)">{desc}</div></div></div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin:16px 0;border-top:1px solid var(--stroke)"></div>', unsafe_allow_html=True)

    # Clear button
    if st.button("🗑️  Clear All Documents", use_container_width=True):
        try:
            httpx.delete(f"{API_URL}/analyze/clear", timeout=5)
            for k in ["analysis_result","chat_history","doc_id","extracted_text","doc_summary_for_questions"]:
                st.session_state[k] = None if k in ["analysis_result","doc_id"] else ([] if k=="chat_history" else "")
            st.success("Cleared!")
            st.rerun()
        except:
            st.error("API not reachable")

    # Active doc badge
    if st.session_state.doc_id:
        st.markdown(f"""<div style="background:rgba(34,211,165,0.07);border:1px solid rgba(34,211,165,0.2);border-radius:10px;padding:10px 12px;margin-top:12px"><div style="font-size:0.68rem;color:#22d3a5;text-transform:uppercase;letter-spacing:0.08em;font-family:'JetBrains Mono',monospace;margin-bottom:3px">Active Document</div><div style="font-size:0.82rem;color:var(--text);font-family:'JetBrains Mono',monospace;word-break:break-all">{st.session_state.doc_id}</div></div>""", unsafe_allow_html=True)

    st.markdown('<div style="margin-top:auto;padding-top:20px;border-top:1px solid var(--stroke);margin-top:24px"><div style="font-size:0.7rem;color:var(--muted);text-align:center">MedAnalyzer AI · v2.0</div></div>', unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# MAIN HEADER — HealthMate style
# ───────────────────────────────────────────────────────────────
# Welcome banner
active_doc  = st.session_state.get("doc_id") or "No document"
doc_label   = doc_type.replace("_"," ").title()
chat_count  = len(st.session_state.get("chat_history",[]))
has_result  = st.session_state.get("analysis_result") is not None

st.markdown(f"""
<div class="welcome-banner" style="margin-top:1.2rem">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px">
    <div>
      <div style="font-size:0.72rem;color:#6ee7cc;text-transform:uppercase;letter-spacing:0.12em;font-family:'JetBrains Mono',monospace;margin-bottom:6px">Clinical Intelligence Workspace</div>
      <div class="welcome-title">Medical AI <span style="color:#29d9c2">Command Center</span></div>
      <div class="welcome-sub">Upload reports · Extract entities · Ask questions · Generate patient plans · Monitor model performance.</div>
      <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
        <span style="background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.22);border-radius:999px;padding:4px 12px;font-size:0.75rem;color:#93c5fd">🧠 AI Summarization</span>
        <span style="background:rgba(34,211,165,0.1);border:1px solid rgba(34,211,165,0.2);border-radius:999px;padding:4px 12px;font-size:0.75rem;color:#6ee7cc">🧬 Medical NER</span>
        <span style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.2);border-radius:999px;padding:4px 12px;font-size:0.75rem;color:#c4b5fd">💬 RAG Q&A</span>
        <span style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.2);border-radius:999px;padding:4px 12px;font-size:0.75rem;color:#fcd34d">📈 MLflow</span>
      </div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap">
      <div style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.18);border-radius:14px;padding:14px 18px;min-width:140px">
        <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;margin-bottom:5px">Doc Type</div>
        <div style="font-size:0.92rem;font-weight:700;color:var(--blue)">{doc_label}</div>
        <div style="font-size:0.72rem;color:var(--muted);margin-top:2px">{"Case loaded" if has_result else "Awaiting upload"}</div>
      </div>
      <div style="background:rgba(34,211,165,0.06);border:1px solid rgba(34,211,165,0.16);border-radius:14px;padding:14px 18px;min-width:140px">
        <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;margin-bottom:5px">Q&A</div>
        <div style="font-size:0.92rem;font-weight:700;color:var(--green)">{chat_count} questions</div>
        <div style="font-size:0.72rem;color:var(--muted);margin-top:2px">{"Interactive" if st.session_state.get("doc_id") else "Locked"}</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Quick stats row (HealthMate style)
analysis_result = st.session_state.get("analysis_result")
if analysis_result:
    stats    = analysis_result.get("entity_stats",{})
    entities = analysis_result.get("entities",{})
    total_e  = sum(stats.values())
    total_d  = len(entities.get("diseases",[]) or [])
    total_dr = len(entities.get("drugs",[]) or [])
    total_l  = len(entities.get("lab_values",[]) or [])
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:0.9rem 0 0.2rem">
      <div class="stat-pill"><div class="stat-icon" style="background:rgba(59,130,246,0.12)">🔬</div><div><div class="stat-val" style="color:var(--blue)">{total_e}</div><div class="stat-lbl">Details Found</div></div></div>
      <div class="stat-pill"><div class="stat-icon" style="background:rgba(239,68,68,0.12)">🩺</div><div><div class="stat-val" style="color:var(--red)">{total_d}</div><div class="stat-lbl">Conditions</div></div></div>
      <div class="stat-pill"><div class="stat-icon" style="background:rgba(245,158,11,0.12)">💊</div><div><div class="stat-val" style="color:var(--amber)">{total_dr}</div><div class="stat-lbl">Medicines</div></div></div>
      <div class="stat-pill"><div class="stat-icon" style="background:rgba(41,217,194,0.12)">🧪</div><div><div class="stat-val" style="color:var(--cyan)">{total_l}</div><div class="stat-lbl">Lab Signals</div></div></div>
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────────────────────
# TABS
# ───────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📄  Analyze Document",
    "💬  Ask Questions",
    "🩺  Report Dashboard",
    "📅  Patient Plan",
    "📊  Performance"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — ANALYZE
# ══════════════════════════════════════════════════════════════
with tab1:
    col_in, col_out = st.columns([1,1], gap="large")
    with col_in:
        st.markdown('<div class="sec-title">Input Document</div>', unsafe_allow_html=True)
        method = st.radio("Input Method", ["✏️ Paste Text","📎 Upload File"], horizontal=True, label_visibility="collapsed")
        text_to_analyze = ""
        if method == "✏️ Paste Text":
            text_to_analyze = st.text_area("Document Text", height=320, placeholder="Paste medical report, lab results, discharge summary, prescription, or clinical document here...", label_visibility="collapsed", key="paste_input")
        else:
            uploaded = st.file_uploader("Upload File", type=["pdf","txt"], label_visibility="collapsed")
            if uploaded:
                if uploaded.name.endswith(".txt"):
                    text_to_analyze = uploaded.read().decode("utf-8","ignore")
                    st.markdown(f'<div class="card card-green" style="padding:8px 14px;font-size:0.83rem">✅ Loaded <b>{len(text_to_analyze):,}</b> chars from <b>{uploaded.name}</b></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="card card-blue" style="padding:8px 14px;font-size:0.83rem">📄 <b>{uploaded.name}</b> — will be processed by API</div>', unsafe_allow_html=True)
                    st.session_state["_pdf_file"] = uploaded

        st.markdown("<div style='margin-top:0.75rem'></div>", unsafe_allow_html=True)
        col_btn1, col_btn2 = st.columns([3,1])
        with col_btn1:
            analyze_clicked = st.button("🔬 Analyze Document", use_container_width=True, key="analyze_btn", type="primary")
        with col_btn2:
            clear_input = st.button("✕ Clear", use_container_width=True, key="clear_btn")
        if clear_input:
            st.session_state.analysis_result = None; st.rerun()
        if analyze_clicked:
            if not api_ok:
                st.error("❌ API is offline. Start the backend first.")
            else:
                with st.spinner("🧬 Running NER + Summarization..."):
                    try:
                        if method == "📎 Upload File" and st.session_state.get("_pdf_file"):
                            f = st.session_state["_pdf_file"]; f.seek(0)
                            resp = httpx.post(f"{API_URL}/analyze/upload", files={"file":(f.name,f.read(),"application/pdf")}, params={"doc_type":doc_type}, timeout=120)
                        else:
                            if not text_to_analyze.strip(): st.error("Please enter or upload a document first."); st.stop()
                            resp = httpx.post(f"{API_URL}/analyze/text", json={"text":text_to_analyze,"doc_type":doc_type}, timeout=120)
                        if resp.status_code == 200:
                            res = resp.json()
                            st.session_state.analysis_result = res
                            st.session_state.doc_id = res.get("doc_id")
                            st.session_state.doc_summary_for_questions = res.get("summary",{}).get("summary","")
                            st.session_state.doc_entities_for_questions = res.get("entities",{})
                            st.session_state.doc_summary_data_for_questions = res.get("summary",{})
                            lat = res.get("latency_ms",0)
                            st.markdown(f'<div class="card card-green" style="padding:8px 14px;font-size:0.83rem">✅ Complete in <b style="color:#22d3a5">{lat/1000:.1f}s</b> · ID: <code style="background:rgba(255,255,255,0.06);padding:2px 6px;border-radius:4px">{res.get("doc_id")}</code></div>', unsafe_allow_html=True)
                        else:
                            try: error_detail = resp.json().get("detail", resp.text)
                            except: error_detail = resp.text
                            st.error(f"Analysis failed: {error_detail}")
                    except Exception as e:
                        st.error(f"Connection failed: {e}")

    with col_out:
        if st.session_state.analysis_result:
            render_structured_report(st.session_state.analysis_result)
        else:
            st.markdown("""
            <div class="card card-blue" style="min-height:420px;display:flex;align-items:center;justify-content:center;text-align:center">
              <div>
                <div style="width:68px;height:68px;background:rgba(59,130,246,0.12);border-radius:20px;display:flex;align-items:center;justify-content:center;font-size:30px;margin:0 auto 16px">🧬</div>
                <div style="font-size:1.1rem;font-weight:700;color:var(--text)">Ready for a medical case</div>
                <div style="font-size:0.88rem;line-height:1.8;color:var(--muted);margin-top:8px;max-width:280px">Upload a report or paste text to unlock AI analysis, Q&A, and patient planning.</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — CHAT / Q&A
# ══════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state.doc_id:
        st.markdown("""
        <div class="card card-blue" style="text-align:center;padding:3rem 2rem">
          <div style="width:60px;height:60px;background:rgba(59,130,246,0.12);border-radius:18px;display:flex;align-items:center;justify-content:center;font-size:28px;margin:0 auto 16px">💬</div>
          <div style="font-size:1rem;font-weight:700;color:var(--text)">Analyze a document first</div>
          <div style="font-size:0.85rem;color:var(--muted);margin-top:6px">Go to <b style="color:var(--blue)">Analyze Document</b> and submit a medical file to unlock Q&A</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="card card-green" style="padding:10px 16px;margin-bottom:1rem;display:flex;align-items:center;gap:10px">
          <div style="width:32px;height:32px;background:rgba(34,211,165,0.15);border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0">📄</div>
          <div>
            <div style="font-size:0.68rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.08em;font-family:'JetBrains Mono',monospace">Active Document</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:0.85rem;color:#22d3a5">{st.session_state.doc_id}</div>
          </div>
          <div style="margin-left:auto;background:rgba(34,211,165,0.1);border:1px solid rgba(34,211,165,0.22);border-radius:999px;padding:4px 12px;font-size:0.76rem;color:#6ee7cc">Q&A Ready</div>
        </div>
        """, unsafe_allow_html=True)

        entities    = st.session_state.get("doc_entities_for_questions",{})
        summ        = st.session_state.get("doc_summary_for_questions","")
        summary_data= st.session_state.get("doc_summary_data_for_questions",{})
        smart_qs    = generate_smart_questions(entities, summ, doc_type, summary_data)

        st.markdown('<div class="sec-title">Quick Questions</div>', unsafe_allow_html=True)
        q_cols = st.columns(3)
        for i, q in enumerate(smart_qs):
            with q_cols[i % 3]:
                if st.button(q, key=f"qq_{i}", use_container_width=True):
                    st.session_state.q_input = q
                    st.session_state.auto_send_question = True
                    st.rerun()

        st.markdown("---")
        st.markdown('<div class="sec-title">Ask Your Own Question</div>', unsafe_allow_html=True)
        user_q = st.text_input("Question", placeholder="e.g. What are the critical lab findings? What medications are prescribed?", label_visibility="collapsed", key="q_input")
        sc1, sc2 = st.columns([3,1])
        with sc1:
            send = st.button("🚀 Send Question", use_container_width=True, key="send_btn", type="primary")
        with sc2:
            if st.button("🗑️ Clear Chat", use_container_width=True, key="clearchat_btn"):
                st.session_state.chat_history = []; st.session_state.auto_send_question = False; st.rerun()

        auto_send = st.session_state.get("auto_send_question", False)
        if (send or auto_send) and user_q.strip():
            st.session_state.auto_send_question = False
            with st.spinner("🔍 Searching with RAG..."):
                try:
                    resp = httpx.post(f"{API_URL}/chat/query", json={"question":user_q,"doc_id":st.session_state.doc_id}, timeout=120)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.chat_history.append({"q":user_q,"a":data["answer"],"confidence":data["confidence"],"latency_ms":data["latency_ms"],"sources":data.get("sources",[])})
                        st.rerun()
                    else:
                        try: st.warning(resp.json().get("detail", resp.text))
                        except: st.warning(resp.text)
                except Exception as e:
                    st.error(f"Connection error: {e}")

        if st.session_state.chat_history:
            st.markdown('<div class="sec-title">Conversation</div>', unsafe_allow_html=True)
            for turn in reversed(st.session_state.chat_history):
                conf = turn["confidence"]
                cc   = "#22d3a5" if conf > 0.7 else "#f59e0b" if conf > 0.4 else "#ef4444"
                st.markdown(f'<div class="bubble-user">🧑 {turn["q"]}</div>', unsafe_allow_html=True)
                st.markdown(f"""<div class="bubble-bot"><div>🤖 {turn["a"]}</div><div class="bubble-meta"><span style="color:{cc}">● {int(conf*100)}% confidence</span> &nbsp;·&nbsp; {turn["latency_ms"]/1000:.1f}s &nbsp;·&nbsp; {len(turn.get("sources",[]))} sources</div></div>""", unsafe_allow_html=True)
                if turn.get("sources"):
                    with st.expander(f"📚 View {len(turn['sources'])} source excerpts"):
                        for j, src in enumerate(turn["sources"],1):
                            meta = src.get("metadata",{}); chunk = meta.get("chunk",j); dt = meta.get("doc_type","")
                            st.markdown(f"""<div style="background:var(--bg3);border:1px solid var(--stroke);border-radius:10px;padding:10px;margin:6px 0;font-size:0.82rem"><div style="color:var(--muted);font-size:0.68rem;margin-bottom:4px;font-family:'JetBrains Mono',monospace">CHUNK {chunk} · {dt.upper()}</div><div style="color:var(--muted);line-height:1.6">{src.get("text","")}</div></div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — REPORT DASHBOARD
# ══════════════════════════════════════════════════════════════
with tab3:
    if st.session_state.analysis_result:
        render_patient_care_dashboard(st.session_state.analysis_result, st.session_state.doc_id, doc_type)
    else:
        st.markdown("""
        <div class="card card-blue" style="text-align:center;padding:3rem 2rem">
          <div style="width:60px;height:60px;background:rgba(59,130,246,0.12);border-radius:18px;display:flex;align-items:center;justify-content:center;font-size:28px;margin:0 auto 16px">🩺</div>
          <div style="font-size:1rem;font-weight:700">Dashboard unlocks after analysis</div>
          <div style="font-size:0.86rem;color:var(--muted);margin-top:6px">Analyze a medical report first to see findings, entities, risk areas, and next steps.</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — PATIENT PLAN
# ══════════════════════════════════════════════════════════════
with tab4:
    if st.session_state.analysis_result:
        render_patient_plan(st.session_state.analysis_result)
    else:
        st.markdown("""
        <div class="card card-blue" style="text-align:center;padding:3rem 2rem">
          <div style="width:60px;height:60px;background:rgba(59,130,246,0.12);border-radius:18px;display:flex;align-items:center;justify-content:center;font-size:28px;margin:0 auto 16px">📅</div>
          <div style="font-size:1rem;font-weight:700">Patient plan will appear here</div>
          <div style="font-size:0.86rem;color:var(--muted);margin-top:6px">After analysis, get a daily recovery plan with routines, diet, exercises, smart tracking, and weekly outlook.</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — PERFORMANCE / MLFLOW
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-title">Performance Dashboard</div>', unsafe_allow_html=True)
    runs_data = []
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient(tracking_uri=MLFLOW_URI)
        experiments = client.search_experiments()
        med_exp = next((e for e in experiments if "medical" in e.name.lower()), None)
        if med_exp:
            runs = client.search_runs(experiment_ids=[med_exp.experiment_id], max_results=50, order_by=["start_time DESC"])
            for run in runs:
                m = run.data.metrics; t = run.data.tags
                runs_data.append({"run_id":run.info.run_id[:8],"name":run.info.run_name,"status":run.info.status,"start":datetime.fromtimestamp(run.info.start_time/1000).strftime("%H:%M:%S"),"latency_ms":m.get("latency_ms",0),"total_entities":m.get("total_entities",0),"diseases_found":m.get("diseases_found",0),"drugs_found":m.get("drugs_found",0),"symptoms_found":m.get("symptoms_found",0),"compression_ratio":m.get("compression_ratio",0),"confidence":m.get("confidence",0),"text_length_words":m.get("text_length_words",0),"doc_type":t.get("doc_type",""),"type":"analysis" if "analysis" in run.info.run_name else "query"})
    except Exception as e:
        st.warning(f"Could not connect to MLflow at {MLFLOW_URI}: {e}")

    if not runs_data:
        ca, cb = st.columns(2)
        with ca:
            st.markdown("""<div class="card card-blue" style="padding:1.2rem"><div style="font-size:0.72rem;color:var(--blue);text-transform:uppercase;letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;margin-bottom:8px">MLflow Server</div><div style="font-size:0.84rem;color:var(--muted);line-height:1.8">Start with:<br><code style="background:rgba(0,0,0,0.3);padding:4px 10px;border-radius:6px;color:#22d3a5;font-family:'JetBrains Mono',monospace">mlflow ui --port 5000</code><br><br>Then open <a href="http://localhost:5000" style="color:var(--blue)">localhost:5000 ↗</a></div></div>""", unsafe_allow_html=True)
        with cb:
            st.markdown("""<div class="card card-green" style="padding:1.2rem"><div style="font-size:0.72rem;color:var(--green);text-transform:uppercase;letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;margin-bottom:8px">What Gets Tracked</div><div style="font-size:0.84rem;color:var(--muted);line-height:2"><span style="color:var(--green)">✓</span> Analysis latency (ms)<br><span style="color:var(--green)">✓</span> Entity counts by type<br><span style="color:var(--green)">✓</span> Compression ratio<br><span style="color:var(--green)">✓</span> RAG confidence scores<br><span style="color:var(--green)">✓</span> Query latency<br><span style="color:var(--green)">✓</span> Document types</div></div>""", unsafe_allow_html=True)
        st.info("No MLflow runs found. Analyze some documents to populate the dashboard.")
    else:
        df = pd.DataFrame(runs_data)
        analysis_df = df[df["type"]=="analysis"]; query_df = df[df["type"]=="query"]
        total_runs = len(df)
        avg_lat    = df["latency_ms"].mean()/1000 if not df.empty else 0
        avg_conf   = query_df["confidence"].mean()*100 if not query_df.empty else 0
        total_ents = analysis_df["total_entities"].sum() if not analysis_df.empty else 0

        st.markdown(f"""<div class="metric-grid">
          <div class="metric-box"><div class="metric-num blue">{total_runs}</div><div class="metric-lbl">Total Activity</div></div>
          <div class="metric-box"><div class="metric-num green">{avg_lat:.1f}s</div><div class="metric-lbl">Avg Response Time</div></div>
          <div class="metric-box"><div class="metric-num orange">{avg_conf:.0f}%</div><div class="metric-lbl">Avg Confidence</div></div>
          <div class="metric-box"><div class="metric-num purple">{int(total_ents)}</div><div class="metric-lbl">Entities Extracted</div></div>
        </div>""", unsafe_allow_html=True)

        overview_parts = [f"{len(analysis_df)} analyses" if len(analysis_df) else None, f"{len(query_df)} Q&A" if len(query_df) else None]
        if not query_df.empty: overview_parts.append(f"avg confidence {avg_conf:.0f}%")
        if not analysis_df.empty: overview_parts.append(f"avg {analysis_df['latency_ms'].mean()/1000:.1f}s processing")
        st.markdown(f"""<div class="card card-blue" style="padding:1rem 1.2rem;margin-bottom:0.8rem"><div style="font-size:0.7rem;color:var(--blue);text-transform:uppercase;letter-spacing:0.1em;font-family:'JetBrains Mono',monospace;margin-bottom:5px">Session Summary</div><div style="font-size:0.92rem;color:var(--text);line-height:1.8">{", ".join([p for p in overview_parts if p])}.</div></div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            if not analysis_df.empty:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=list(range(len(analysis_df))), y=analysis_df["latency_ms"].values/1000,
                    mode="lines+markers", line=dict(color="#3b82f6",width=2), marker=dict(size=6,color="#3b82f6"), name="Latency (s)"))
                fig.update_layout(title=dict(text="Processing Time",font=dict(size=12,color="#7a95b8")),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#7a95b8",family="Inter"), height=240,
                    xaxis=dict(gridcolor="rgba(255,255,255,0.04)",title="Analyses"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.04)",title="Seconds"),
                    margin=dict(t=35,b=30,l=40,r=10))
                st.plotly_chart(fig, use_container_width=True)
        with c2:
            if not analysis_df.empty:
                ent_vals = [analysis_df[c].sum() for c in ["diseases_found","drugs_found","symptoms_found"]]
                fig2 = go.Figure(go.Pie(labels=["Diseases","Drugs","Symptoms"], values=ent_vals,
                    marker_colors=["#fca5a5","#93c5fd","#fcd34d"], hole=0.45, textfont=dict(color="#7a95b8")))
                fig2.update_layout(title=dict(text="Entity Breakdown",font=dict(size=12,color="#7a95b8")),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#7a95b8",family="Inter"), height=240,
                    margin=dict(t=35,b=10,l=10,r=10), legend=dict(font=dict(color="#7a95b8")))
                st.plotly_chart(fig2, use_container_width=True)

        if not query_df.empty:
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(x=list(range(len(query_df))), y=query_df["confidence"].values,
                marker_color=["#22d3a5" if c>0.7 else "#f59e0b" if c>0.4 else "#ef4444" for c in query_df["confidence"].values],
                text=[f"{c:.0%}" for c in query_df["confidence"].values], textposition="outside", textfont=dict(color="#7a95b8")))
            fig3.update_layout(title=dict(text="Answer Confidence",font=dict(size=12,color="#7a95b8")),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#7a95b8",family="Inter"), height=210,
                yaxis=dict(range=[0,1.1],gridcolor="rgba(255,255,255,0.04)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.04)",title="Recent Q&A"),
                margin=dict(t=35,b=30,l=40,r=10))
            st.plotly_chart(fig3, use_container_width=True)

        st.markdown('<div class="sec-title">Recent Activity</div>', unsafe_allow_html=True)
        st.markdown('<div style="background:var(--panel);border:1px solid var(--stroke);border-radius:12px;padding:10px 14px;margin-bottom:8px;font-size:0.81rem;color:var(--muted);line-height:1.6">Activity log of recent analyses and Q&A sessions. Not a medical report.</div>', unsafe_allow_html=True)
        activity_df = df[["type","start","latency_ms","total_entities","confidence","doc_type"]].copy()
        activity_df["Action"]        = activity_df["type"].apply(lambda v: "Analyzed a document" if v=="analysis" else "Answered a question")
        activity_df["When"]          = activity_df["start"]
        activity_df["Time Taken"]    = (activity_df["latency_ms"]/1000).round(2).astype(str) + " sec"
        activity_df["Document Type"] = activity_df["doc_type"].replace("","General").str.replace("_"," ").str.title()
        activity_df["Result"]        = activity_df.apply(lambda row: f"Found {int(row['total_entities'])} details" if row["type"]=="analysis" else f"Answered with {int(row['confidence']*100)}% confidence", axis=1)
        st.dataframe(activity_df[["Action","When","Time Taken","Document Type","Result"]], use_container_width=True, hide_index=True)
        st.markdown(f"<div style='text-align:right;margin-top:6px'><a href='http://localhost:5000' target='_blank' style='color:var(--blue);font-size:0.8rem;text-decoration:none'>→ Open MLflow UI ↗</a></div>", unsafe_allow_html=True)