"""
ResumeReviewer-Agent — Streamlit App
Run: streamlit run app.py
"""

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Reviewer Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# History file (mirrors notebook's resume_analysis_history.txt)
HISTORY_FILE = Path("resume_analysis_history.txt")


# ══════════════════════════════════════════════════════════════════════════════
# Pydantic Schema
# ══════════════════════════════════════════════════════════════════════════════

class JobTarget(BaseModel):
    company_name: str = Field(
        description="Name of the company hiring, or 'Unknown' if not mentioned in the posting."
    )
    job_title: str = Field(
        description="Target job title extracted directly from the posting."
    )


class ActionableRewrite(BaseModel):
    current_text: str = Field(
        description="Weak or generic phrasing currently found on the resume."
    )
    suggested_text: str = Field(
        description="Quantified, metric-driven rewrite tailored to real JD requirements."
    )
    reasoning: str = Field(
        description="Why this specific revision increases relevance for this role."
    )


class ResumeAnalysisReport(BaseModel):
    extracted_jobs: List[JobTarget] = Field(
        description="All job titles and companies identified directly from the job postings."
    )
    matching_skills: List[str] = Field(
        description="Key technical skills present in both the resume and live job postings."
    )
    missing_skills: List[str] = Field(
        description="Critical technical skills demanded by job market data but missing from resume."
    )
    skill_gap_summary: str = Field(
        description="High-level assessment of candidate readiness and alignment."
    )
    bullet_rewrites: List[ActionableRewrite] = Field(
        description="3 to 5 concrete resume bullet point improvements."
    )


# ══════════════════════════════════════════════════════════════════════════════
# Backend helpers  (adapted from notebook)
# ══════════════════════════════════════════════════════════════════════════════

def load_resume_text(file_path: str) -> str:
    """Parses a multi-page PDF resume or docx resume into a consolidated text string."""
    if file_path.endswith(".pdf"):
        from langchain_community.document_loaders import PyMuPDFLoader

        loader = PyMuPDFLoader(file_path)
        pages = loader.load()
        raw_text = "\n\n".join([page.page_content for page in pages])
        resume_context = re.sub(r"\xa0", " ", raw_text)
        return resume_context

    elif file_path.endswith((".doc", ".docx")):
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader

        loader = UnstructuredWordDocumentLoader(file_path)
        pages = loader.load()
        raw_text = "\n\n".join([page.page_content for page in pages])
        resume_context = re.sub(r"\xa0", " ", raw_text)
        resume_context = re.sub(r"\n\s*\n", "\n\n", resume_context)
        return resume_context

    else:
        raise ValueError(
            f"Unsupported file extension for '{file_path}'. "
            "Only .pdf, .doc, and .docx formats are supported."
        )


def load_and_clean_job_descriptions(
    text: str,
    remove_patterns: list | None = None,
) -> str:
    """
    Accepts raw job description text (already read from file or pasted by user),
    splits by '------' separator, and cleans whitespace — mirrors notebook logic.
    """
    if remove_patterns is None:
        remove_patterns = [
            "Save",
            "Apply",
            "Easy Apply",
            "No response insights available yet",
            "Your profile and resume match the required qualifications well",
            "See how you compare to over 100 other applicants",
            "See how you compare to over 100 others who clicked apply",
            "BETA • Is this information helpful?",
            "Restart Premium for €0",
            "Access exclusive applicant insights, see jobs where you have the highest chance of hearing back, and more.",
            "Show match details",
            "Responses managed off LinkedIn",
            "Message",
        ]

    # Phase 1: split & collapse whitespace
    raw_jobs = text.split("------")
    cleaned_jobs = []
    for i, job in enumerate(raw_jobs, 1):
        job = job.strip()
        if not job:
            continue
        job = re.sub(r"\n\s*\n+", "\n\n", job)
        cleaned_jobs.append(f"====== JOB POSTING {i} ======\n{job}")

    formatted = "\n\n".join(cleaned_jobs)

    # Phase 2: remove boilerplate lines
    lines = formatted.splitlines()
    filtered = [
        line
        for line in lines
        if not any(p.lower() == line.lower() for p in remove_patterns)
    ]
    formatted = re.sub(r"\n\s*\n+", "\n\n", "\n".join(filtered))

    return formatted


@st.cache_resource
def get_chain():
    """Builds and caches the LCEL prompt | structured_llm chain."""
    system_prompt = (
        "You are a Senior Technical Recruiter and AI Career Coach specializing in "
        "Machine Learning, Data Science, and AI engineering roles.\n"
        "Your task is to analyze a candidate's resume against local target job descriptions.\n\n"
        "First, identify all job titles and company names present in the job descriptions.\n"
        "Then, compare the candidate's resume against those requirements and generate a "
        "structured analysis report matching the JSON schema."
    )

    user_prompt = (
        "CANDIDATE RESUME:\n{resume_context}\n\n"
        "TARGET JOB POSTINGS:\n{job_data}\n"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", user_prompt)]
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    structured_llm = llm.with_structured_output(ResumeAnalysisReport)
    return prompt | structured_llm


def format_report_text(report: ResumeAnalysisReport) -> str:
    """Renders a ResumeAnalysisReport as a human-readable string (mirrors notebook)."""
    run_id = f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    job_targets_formatted = "\n".join(
        f"- {j.company_name} - {j.job_title}" for j in report.extracted_jobs
    )

    body = (
        f"Matching Skills:\n"
        f"{', '.join(report.matching_skills)}\n"
        f"------------------------------------------------\n"
        f"Missing Skills:\n"
        f"{', '.join(report.missing_skills)}\n"
        f"------------------------------------------------\n"
        f"Summary:\n"
        f"{report.skill_gap_summary}\n"
        f"------------------------------------------------\n"
        f"Bullet Rewrites:\n"
    )

    for idx, rw in enumerate(report.bullet_rewrites, 1):
        body += (
            f"\n{idx}. Current: {rw.current_text}\n"
            f"   Suggested: {rw.suggested_text}\n"
            f"   Reasoning: {rw.reasoning}\n"
        )

    return (
        f"{run_id}\n{job_targets_formatted}\n\n"
        f"Full ResumeReport text:\n{body.strip()}\n"
        f"================================================\n\n"
    )


def save_to_history(report_text: str) -> None:
    """Appends the formatted report to the history log file."""
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        f.write(report_text)


# ══════════════════════════════════════════════════════════════════════════════
# UI — Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("⚙️ Configuration")
    st.divider()

    # Resume upload
    st.subheader("📄 Resume")
    resume_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "docx", "doc"],
        help="Supported formats: PDF, DOCX, DOC",
    )

    st.divider()

    # Job description
    st.subheader("💼 Job Description(s)")
    jd_input_mode = st.radio(
        "Input method",
        options=["Upload .txt file", "Paste text"],
        horizontal=True,
    )

    job_desc_raw: str | None = None

    if jd_input_mode == "Upload .txt file":
        jd_file = st.file_uploader(
            "Upload job_desc.txt",
            type=["txt"],
            help="Separate multiple job postings with '------'",
        )
        if jd_file is not None:
            job_desc_raw = jd_file.read().decode("utf-8")
    else:
        pasted = st.text_area(
            "Paste job description(s) here",
            height=280,
            placeholder=(
                "Paste one or more job descriptions.\n"
                "Separate multiple postings with a line containing only: ------"
            ),
        )
        if pasted.strip():
            job_desc_raw = pasted

    st.divider()

    # Analyze button
    analyze_btn = st.button(
        "🔍 Analyze Resume",
        type="primary",
        use_container_width=True,
        disabled=(resume_file is None or job_desc_raw is None),
    )

    if resume_file is None:
        st.caption("⚠️ Upload a resume to continue.")
    if job_desc_raw is None:
        st.caption("⚠️ Provide job description(s) to continue.")


# ══════════════════════════════════════════════════════════════════════════════
# UI — Main Dashboard
# ══════════════════════════════════════════════════════════════════════════════

st.title("📄 Resume Reviewer Agent")
st.markdown(
    "Analyze your resume against job descriptions using a structured AI pipeline "
    "powered by Gemini + LangChain."
)
st.divider()

# Session state
if "report" not in st.session_state:
    st.session_state.report = None
if "report_text" not in st.session_state:
    st.session_state.report_text = None

# ── Run analysis ──────────────────────────────────────────────────────────────
if analyze_btn:
    with st.spinner("🤖 Analyzing your resume against the job postings…"):
        try:
            # 1. Save uploaded resume to a temp file so LangChain loaders can open it
            suffix = Path(resume_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(resume_file.read())
                tmp_path = tmp.name

            resume_context = load_resume_text(tmp_path)
            os.unlink(tmp_path)

            # 2. Clean job description text
            job_data = load_and_clean_job_descriptions(job_desc_raw)

            # 3. Invoke LCEL chain
            chain = get_chain()
            report: ResumeAnalysisReport = chain.invoke(
                {"resume_context": resume_context, "job_data": job_data}
            )

            # 4. Format & persist to history file
            report_text = format_report_text(report)
            save_to_history(report_text)

            st.session_state.report = report
            st.session_state.report_text = report_text

        except Exception as exc:
            st.error(f"❌ Analysis failed: {exc}")
            st.stop()

# ── Render report ─────────────────────────────────────────────────────────────
if st.session_state.report is not None:
    report: ResumeAnalysisReport = st.session_state.report
    report_text: str = st.session_state.report_text

    # Header row: jobs targeted + download button
    header_col, dl_col = st.columns([3, 1], vertical_alignment="bottom")

    with header_col:
        st.subheader("🎯 Jobs Analyzed")
        job_labels = [
            f"{j.company_name} — {j.job_title}" for j in report.extracted_jobs
        ]
        st.pills("Positions", job_labels, default=job_labels, disabled=True)

    with dl_col:
        st.download_button(
            label="⬇️ Download Report (.txt)",
            data=report_text,
            file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    st.divider()

    # Two-column layout: skills | summary
    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        with st.expander("✅ Matching Skills", expanded=True):
            for skill in report.matching_skills:
                st.markdown(f"- {skill}")

        with st.expander("❌ Missing Skills", expanded=True):
            for skill in report.missing_skills:
                st.markdown(f"- {skill}")

    with right_col:
        st.subheader("🧠 Skill Gap Summary")
        st.info(report.skill_gap_summary)

    st.divider()

    # Bullet rewrites (full width)
    st.subheader("✏️ Bullet Point Rewrites")

    for idx, rw in enumerate(report.bullet_rewrites, 1):
        with st.expander(f"Rewrite #{idx}", expanded=True):
            before_col, after_col = st.columns(2, gap="medium")

            with before_col:
                st.markdown("**Before**")
                st.warning(rw.current_text)

            with after_col:
                st.markdown("**After**")
                st.success(rw.suggested_text)

            st.markdown("**💡 Reasoning**")
            st.caption(rw.reasoning)

    st.divider()

    # Raw report text (collapsible)
    with st.expander("📋 Full Report Text", expanded=False):
        st.code(report_text, language=None)

    st.caption(
        f"Report saved to `{HISTORY_FILE.resolve()}` · "
        f"Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

else:
    # Placeholder when no report yet
    st.markdown(
        """
        <div style="text-align:center; padding: 4rem 0; color: #888;">
            <h3>👈 Configure inputs in the sidebar and click <em>Analyze Resume</em></h3>
            <p>Your structured analysis will appear here.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
