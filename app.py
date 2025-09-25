import os
import re
from datetime import datetime
from typing import Dict, List

import streamlit as st

# Import local modules
from src.helper import init_pinecone, get_embedding_model, get_chat_model
from src.pdf_utils import process_coursebook_pdf, process_patient_pdf
from src.rag import build_retrievers, build_rag_chain, ask

# ================================
# Page Setup & Directories
# ================================

APP_TITLE = "🏥 First Aid Medical Chatbot"
COURSE_DIR = "data/medical_course"
PATIENT_DIR = "data/patient_data"
COURSE_NAMESPACE = "Medical_Course"

st.set_page_config(page_title=APP_TITLE, layout="wide")

def ensure_dirs():
    os.makedirs(COURSE_DIR, exist_ok=True)
    os.makedirs(PATIENT_DIR, exist_ok=True)

ensure_dirs()

DEFAULT_SAMPLE_PATIENT_URL = os.getenv("SAMPLE_PATIENT_URL")

# ================================
# Initialize Backends
# ================================

@st.cache_resource(show_spinner=False)
def _bootstrap_backends():
    pc, index_name = init_pinecone()
    embedding = get_embedding_model()
    llm = get_chat_model()
    return pc, index_name, embedding, llm

pc, index_name, embedding, llm = _bootstrap_backends()

# ================================
# Init Session State
# ================================

_defaults = {
    "current_patient": "None",
    "chat_history": {},
    "uploaded_courses": set(),
    "message_count": 0,
    "patient_uploader_key": 0,
    "course_uploader_key": 0,
    # Dialog flags + meta
    "show_patient_dialog": False,
    "processing_patient": False,
    "patient_meta": None,
    "show_course_dialog": False,
    "processing_course": False,
    "course_meta": None,
    # Patient list UX improvements
    "patients_cache": set(),     # local cache so dropdown updates immediately
    "patient_selector_key": 0,   # force selectbox to re-render when we add a new patient
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ================================
# Helpers
# ================================

def highlight_medical_terms(text: str) -> str:
    terms = [
        "epinephrine", "adrenaline", "aspirin", "ibuprofen", "paracetamol", "acetaminophen",
        "CPR", "cardiopulmonary resuscitation", "Heimlich", "tourniquet",
        "shock", "anaphylaxis", "asthma", "stroke", "burn", "fracture",
        "airway", "breathing", "circulation", "defibrillator", "AED",
        "bleeding", "poisoning", "choking", "seizure",
    ]

    def repl(m):
        return (
            "<span style='background:#fffae6;border:1px solid #ffe58f;"
            "border-radius:6px;padding:0 4px;'>" + m.group(0) + "</span>"
        )

    for t in sorted(terms, key=len, reverse=True):
        text = re.sub(rf"(?i)\\b{re.escape(t)}\\b", repl, text)
    return text

def icon_for_namespace(ns: str) -> str:
    return "📖" if ns == COURSE_NAMESPACE else "🏥"

def pretty_source_label(src_path: str) -> str:
    base = os.path.basename(src_path)
    base = re.sub(r"\\.pdf$", "", base, flags=re.IGNORECASE)
    return base

def build_clickable_citations(sources: List[Dict], turn_idx: int):
    if not sources:
        return "", []
    lines = []
    for i, s in enumerate(sources, start=1):
        ns = s.get("namespace") or ""
        src = s.get("source") or ""
        page = s.get("page")
        chunk = s.get("chunk", "")
        try:
            page_int = int(page)
        except Exception:
            page_int = None
        icon = icon_for_namespace(ns if ns else COURSE_NAMESPACE)
        ns_label = ns if ns else (COURSE_NAMESPACE if "medical_course" in src.lower() else "Patient")
        page_str = f" — page {page_int}" if page_int is not None else ""
        src_label = pretty_source_label(src) if src else ns_label
        anchor = f"src-{turn_idx}-{i}"
        header = f"<div id='{anchor}'>[{i}] {icon} **{ns_label}** — {src_label}{page_str}</div>"
        expander = (
            "<details><summary>Show context</summary>"
            "<div style='white-space:pre-wrap;font-size:smaller;background:#fafafa;"
            "border:1px solid #ddd;padding:6px;border-radius:6px;margin-top:4px;'>"
            + chunk + "</div></details>" if chunk else ""
        )
        lines.append(header + expander)
    citation_html = " " + " ".join(
        f"<a href='#src-{turn_idx}-{i}' target='_self'>[{i}]</a>"
        for i in range(1, len(sources) + 1)
    )
    return citation_html, lines

def list_patient_namespaces():
    try:
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        return [ns for ns in namespaces if ns != COURSE_NAMESPACE]
    except Exception:
        return []

# ================================
# Dialogs (container-based)
# ================================

def patient_dialog():
    save_path, patient_id, filename = st.session_state.patient_meta

    progress = st.progress(0)
    st.write("Step 1: Splitting into chunks…")
    progress.progress(30)

    st.write("Step 2: Uploading to Pinecone…")
    process_patient_pdf(save_path, patient_id, embedding, pc, index_name)
    progress.progress(90)

    st.success(f"✅ Done! Patient file '{filename}' uploaded.")
    progress.progress(100)

    if st.button("Close ✅", key="close_patient"):
        st.session_state.patients_cache.add(patient_id)
        st.session_state.current_patient = patient_id
        st.session_state.show_patient_dialog = False
        st.session_state.processing_patient = False
        st.session_state.patient_uploader_key += 1
        st.session_state.patient_selector_key += 1
        st.rerun()

def coursebook_dialog():
    save_path, filename = st.session_state.course_meta

    progress = st.progress(0)
    st.write("Step 1: Checking ingestion records…")

    if filename in st.session_state.uploaded_courses:
        progress.progress(100)
        st.success(f"✅ Done! Coursebook '{filename}' already processed.")
    else:
        progress.progress(20)
        st.write("Step 2: Splitting into chunks…")
        progress.progress(50)
        st.write("Step 3: Uploading to Pinecone…")
        process_coursebook_pdf(save_path, embedding, index_name)
        st.session_state.uploaded_courses.add(filename)
        progress.progress(90)
        st.success(f"✅ Done! Coursebook '{filename}' uploaded.")
        progress.progress(100)

    if st.button("Close ✅", key="close_course"):
        st.session_state.show_course_dialog = False
        st.session_state.processing_course = False
        st.session_state.course_uploader_key += 1
        st.rerun()

# ================================
# Sidebar
# ================================

with st.sidebar:
    st.title("⚙️ Controls")

    st.markdown("#### 🔍 Search Mode")
    search_mode = st.radio(
        "Search Mode",
        options=["Both", "Client Only", "Coursebook Only"],
        label_visibility="collapsed"
    )

    remote_patients = sorted(list_patient_namespaces())
    all_patients = sorted(set(remote_patients) | set(st.session_state.patients_cache))
    patients = ["None"] + all_patients

    st.markdown("**Select Client**")
    st.session_state.current_patient = st.selectbox(
        "Select Client",
        options=patients,
        index=patients.index(st.session_state.current_patient) if st.session_state.current_patient in patients else 0,
        format_func=lambda x: ("🏥 " + x) if x != "None" else "None",
        key=f"patient_select_{st.session_state.patient_selector_key}",
        label_visibility="collapsed"
    )

    if st.session_state.current_patient != "None":
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history[st.session_state.current_patient] = []
            st.success("History cleared")

    st.subheader("📥 Upload PDFs")

    patient_pdf = st.file_uploader(
        "Upload Client PDF - Max Size 5 MB", type="pdf", key=f"patient_pdf_{st.session_state.patient_uploader_key}"
    )
    if patient_pdf is not None and not st.session_state.processing_patient:
        filename = patient_pdf.name
        patient_id = os.path.splitext(filename)[0]
        save_path = os.path.join(PATIENT_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(patient_pdf.read())

        st.session_state.patient_meta = (save_path, patient_id, filename)
        st.session_state.show_patient_dialog = True
        st.session_state.processing_patient = True
        st.rerun()

    if st.session_state.show_patient_dialog:
        patient_dialog()

    course_pdf = st.file_uploader(
        "Upload Coursebook PDF  - Max Size 20 MB", type="pdf", key=f"course_pdf_{st.session_state.course_uploader_key}"
    )
    if course_pdf is not None and not st.session_state.processing_course:
        filename = course_pdf.name
        save_path = os.path.join(COURSE_DIR, filename)
        with open(save_path, "wb") as f:
            f.write(course_pdf.read())

        st.session_state.course_meta = (save_path, filename)
        st.session_state.show_course_dialog = True
        st.session_state.processing_course = True
        st.rerun()

    if st.session_state.show_course_dialog:
        coursebook_dialog()

# ================================
# Main Chat Display
# ================================

current = st.session_state.current_patient
if current not in st.session_state.chat_history:
    st.session_state.chat_history[current] = []

st.header("👨 Data Architect's AI Assistant 🧠🔍")

for turn_idx, turn in enumerate(st.session_state.chat_history[current]):
    st.markdown(
        (
            "<div style='text-align:right;background:#e8f9ee;padding:8px;border-radius:8px;"
            "margin:4px 0;'><b>" + turn["question"] + "</b></div>"
        ),
        unsafe_allow_html=True,
    )

    sources = turn.get("sources", [])
    citation_html, source_lines = build_clickable_citations(sources, turn_idx)
    answer_with_cites = (turn.get("answer", "") or "") + citation_html
    answer_html = highlight_medical_terms(answer_with_cites)
    st.markdown(
        (
            "<div style='text-align:left;background:#f2f2f2;padding:8px;border-radius:8px;"
            "margin:4px 0;'>" + answer_html + "</div>"
        ),
        unsafe_allow_html=True,
    )

    if source_lines:
        with st.expander("**Sources**"):
            for line in source_lines:
                st.markdown(line, unsafe_allow_html=True)

# ================================
# Chat Input
# ================================

user_msg = st.chat_input("Ask your medical question…")
if user_msg and user_msg.strip():
    question = user_msg.strip()

    if search_mode == "Client Only":
        retriever = build_retrievers(
            index_name,
            embedding,
            patient_id=current if current != "None" else None,
            course_namespace=None,
        )
    elif search_mode == "Coursebook Only":
        retriever = build_retrievers(
            index_name,
            embedding,
            patient_id=None,
            course_namespace=COURSE_NAMESPACE,
        )
    else:
        retriever = build_retrievers(
            index_name,
            embedding,
            patient_id=current if current != "None" else None,
            course_namespace=COURSE_NAMESPACE,
        )

    chain = build_rag_chain(llm, retriever)

    with st.spinner("Thinking..."):
        result = ask(chain, question)

    sources = result.get("sources", [])
    contexts = result.get("contexts", [])

    for src in sources:
        for ctx in contexts:
            if (
                src.get("source") == ctx.get("source")
                and src.get("page") == ctx.get("page")
                and src.get("namespace") == ctx.get("namespace")
            ):
                src["chunk"] = ctx.get("chunk", "")
                break

    st.session_state.chat_history[current].append(
        {
            "question": question,
            "answer": result.get("answer", ""),
            "sources": sources,
            "timestamp": datetime.now().isoformat(),
        }
    )
    st.rerun()

# ================================
# Footer Disclaimer
# ================================

st.markdown("---")
