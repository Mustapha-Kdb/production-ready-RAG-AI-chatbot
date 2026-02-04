import asyncio
from pathlib import Path
import time
import boto3

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests

load_dotenv()

st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")

# === Futuristic UI styles ===
st.markdown(
    """
    <style>
    /* Base */
    html, body, [data-testid="stAppViewContainer"] {
        background: radial-gradient(1200px 800px at 20% -10%, #1b1f3a 0%, #0b0f1a 45%, #05070d 100%);
        color: #e6f0ff;
        font-family: "Inter", "Segoe UI", system-ui, -apple-system, sans-serif;
    }

    /* Header glow */
    .main .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Neon divider */
    hr {
        border: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #4de2ff, transparent);
        box-shadow: 0 0 12px rgba(77, 226, 255, 0.35);
    }

    /* Glass cards */
    .stForm, .stAlert, .stSuccess, .stInfo, .stWarning, .stError, .stSpinner,
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255, 255, 255, 0.06) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25), 0 0 30px rgba(77, 226, 255, 0.08);
        backdrop-filter: blur(10px);
        border-radius: 14px;
    }

    /* Buttons */
    .stButton > button, .stForm button {
        background: linear-gradient(135deg, #4de2ff, #7d5cff);
        color: #0b0f1a;
        border: none;
        border-radius: 999px;
        padding: 0.6rem 1.3rem;
        font-weight: 600;
        box-shadow: 0 0 18px rgba(125, 92, 255, 0.45);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .stButton > button:hover, .stForm button:hover {
        transform: translateY(-1px);
        box-shadow: 0 0 28px rgba(77, 226, 255, 0.6);
    }

    /* Inputs */
    .stTextInput input, .stNumberInput input, .stTextArea textarea {
        background: rgba(7, 10, 18, 0.7) !important;
        border: 1px solid rgba(77, 226, 255, 0.25) !important;
        color: #e6f0ff !important;
        border-radius: 10px !important;
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        border-style: dashed !important;
        border-color: rgba(77, 226, 255, 0.35) !important;
    }

    /* Titles */
    h1, h2, h3 {
        color: #e6f0ff;
        text-shadow: 0 0 12px rgba(77, 226, 255, 0.35);
        letter-spacing: 0.5px;
    }

    /* Subtle animated aurora */
    .aurora {
        position: fixed;
        z-index: 0;
        width: 1200px;
        height: 800px;
        left: -200px;
        top: -200px;
        background: conic-gradient(from 180deg, #4de2ff, #7d5cff, #4de2ff);
        filter: blur(120px);
        opacity: 0.18;
        animation: spin 18s linear infinite;
        pointer-events: none;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    /* Ensure content sits above aurora */
    .block-container { position: relative; z-index: 1; }
    </style>
    <div class="aurora"></div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def save_uploaded_pdf(file) -> Path:
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise RuntimeError("S3_BUCKET env var is required")
    s3 = boto3.client("s3")
    key = f"uploads/{int(time.time())}_{file.name}"
    s3.upload_fileobj(file, bucket, key)
    return Path(f"s3://{bucket}/{key}")


async def send_rag_ingest_event(pdf_path: Path) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "pdf_path": str(pdf_path.resolve()),
                "source_id": pdf_path.name,
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_pdf(uploaded)
        # Kick off the event and block until the send completes
        asyncio.run(send_rag_ingest_event(path))
        # Small pause for user feedback continuity
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {path.name}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )

    return result[0]


def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = 5
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            # Fire-and-forget event to Inngest for observability/workflow
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            # Poll the local Inngest API for the run's output
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")
