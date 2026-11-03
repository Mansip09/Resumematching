import io
import os
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Optional
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
import pandas as pd
import requests
import streamlit as st
import json

st.set_page_config(page_title="Placement ATS | v2", page_icon="🎓", layout="wide")

# ----------------------------- Styling ---------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1500px;}
.hero {padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(120deg,#14213d,#234e70); color: white; margin-bottom: 1rem;}
.hero h1 {margin: 0; font-size: 2rem;}
.hero p {margin: .45rem 0 0 0; opacity: .9;}
.small-muted {color: #64748b; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>Placement ATS <span style="font-size:1rem;opacity:.8">v2.0</span></h1>
  <p>Resume matching · Student identity verification · CTC eligibility · Excel reporting</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------- Helpers ---------------------------------------
def norm_name(value: object) -> str:
    """Normalize a name for conservative identity comparison."""
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.lower().replace("_", " ").replace("-", " ")
    value = re.sub(r"[^a-z0-9 ]", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def parse_ctc_range(value: object):
    """
    Parse ranges such as '8-12 LPA', '8–12 LPA', '8 to 12 LPA'.
    Returns (minimum, maximum) in LPA, or None if ambiguous/invalid.
    """
    if pd.isna(value):
        return None
    text = str(value).strip().lower().replace(",", "")
    text = text.replace("–", "-").replace("—", "-")
    nums = re.findall(r"\d+(?:\.\d+)?", text)
    if len(nums) != 2:
        return None
    lo, hi = map(float, nums)
    if lo < 0 or hi < lo:
        return None
    return lo, hi

def ctc_eligibility(offered_lpa: float, range_value: object):
    parsed = parse_ctc_range(range_value)
    if parsed is None:
        return "Needs review", "CTC range is missing or could not be parsed"
    lo, hi = parsed
    if lo <= offered_lpa <= hi:
        return "Eligible-needs review", f"Offer {offered_lpa:g} LPA is within {lo:g}–{hi:g} LPA"
    if offered_lpa < lo:
        return "Not eligible", f"Offer is below the student's minimum ({lo:g} LPA)"
    return "Eligible", f"Offer exceeds the student's maximum ({hi:g} LPA)"

def get_column(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    return None

def identity_match(resume_name: str, roster_names):
    """
    Exact normalized name first. A fuzzy match is only a review suggestion;
    it is never silently accepted as verified.
    """
    target = norm_name(resume_name)
    if not target:
        return {"matched_name": "", "status": "Needs review", "confidence": 0.0,
                "note": "No student name could be extracted from the resume"}
    normalized = [(str(n), norm_name(n)) for n in roster_names if norm_name(n)]
    exact = [original for original, normalized_name in normalized if normalized_name == target]
    if len(exact) == 1:
        return {"matched_name": exact[0], "status": "Verified", "confidence": 1.0,
                "note": "Exact normalized name match"}
    if len(exact) > 1:
        return {"matched_name": " / ".join(exact), "status": "Needs review", "confidence": 1.0,
                "note": "Duplicate student names in roster"}
    ranked = sorted(
        [(SequenceMatcher(None, target, normalized_name).ratio(), original)
         for original, normalized_name in normalized],
        reverse=True
    )
    if ranked and ranked[0][0] >= 0.88:
        return {"matched_name": ranked[0][1], "status": "Needs review",
                "confidence": round(ranked[0][0], 3),
                "note": "Possible name variation; confirm manually"}
    return {"matched_name": "", "status": "Unmatched", "confidence": 0.0,
            "note": "No reliable roster identity match"}


def extract_name_from_resume_text(text: str):
    """
    Heuristic only. For reliable identity verification, prefer an explicit
    student ID/email in the roster and resume. Name extraction needs human review.
    """
    for line in text.splitlines()[:12]:
        line = re.sub(r"\s+", " ", line).strip()
        if not line or len(line) > 70:
            continue
        if "@" in line or re.search(r"\b(resume|curriculum vitae|mobile|phone|linkedin|github)\b", line, re.I):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z.'-]*", line)
        if 2 <= len(words) <= 5 and all(len(w) > 1 for w in words):
            if not re.search(r"\b(engineer|developer|analyst|student|manager|intern)\b", line, re.I):
                return line
    return ""

def identity_from_resume_text(text: str, extracted_name: str, roster_names, known_matches=None):
    """Match roster names anywhere in extracted resume text before using a name heuristic."""
    if known_matches is None:
        normalized_text = f" {norm_name(text)} "
        roster_matches = []
        seen = set()
        for name in roster_names:
            normalized_name = norm_name(name)
            if normalized_name and normalized_name not in seen and f" {normalized_name} " in normalized_text:
                roster_matches.append(str(name))
                seen.add(normalized_name)
    else:
        roster_matches = [str(name) for name in known_matches]

    if len(roster_matches) == 1:
        return identity_match(roster_matches[0], roster_names)
    if len(roster_matches) > 1:
        return {
            "matched_name": " / ".join(roster_matches),
            "status": "Needs review",
            "confidence": 0.0,
            "note": "Multiple roster names were found in the resume text",
        }

    return identity_match(extracted_name or extract_name_from_resume_text(text), roster_names)

def extract_text_from_upload(upload):
    name = upload.name.lower()
    raw = upload.getvalue()
    if name.endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        from docx import Document
        doc = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)
    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")
    return ""

def call_backend(base_url, api_key, folder_link, job_description, include_breakdown, ai_feedback, roster_names):
    payload = {
        "folder_link": folder_link,
        "job_description": job_description.strip(),
        "roster_names": roster_names,
        "include_subfolders": True,
        "include_breakdown": include_breakdown,
        "ai_feedback": ai_feedback,
    }

    url = base_url.rstrip("/") + "/analyze-folder"
    st.caption(f"Backend endpoint: {url}")

    response = requests.post(
        url,
        json=payload,
        headers={"X-API-Key": api_key},
        timeout=900,
    )

    if not response.ok:
        st.error(f"FastAPI returned HTTP {response.status_code}")
        st.code(response.text)
        response.raise_for_status()

    return response.json()



def call_text_backend(backend_url, api_key, resume_text, job_description, roster_names, ai_feedback=False, include_breakdown=True):
    url = backend_url.rstrip("/") + "/analyze-text"

    payload = {
        "resume_text": resume_text,
        "job_description": job_description.strip(),
        "roster_names": roster_names,
        "ai_feedback": ai_feedback,
    }

    response = requests.post(
        url,
        json=payload,
        headers={"X-API-Key": api_key},
        timeout=300,
    )

    if not response.ok:
        raise RuntimeError(
            f"Backend returned HTTP {response.status_code}: {response.text}"
        )

    result = response.json()

    if not include_breakdown:
        result.pop("REQUIREMENT_BREAKDOWN", None)

    return result


def _score_value(value):
    """Convert numeric scores and percentage strings to floats."""
    if value in (None, ""):
        return None
    try:
        return float(str(value).strip().removesuffix("%").strip())
    except (TypeError, ValueError):
        return None


def calculate_combined_score(ats_score, semantic_score):
    """Blend ATS (65%) and semantic (35%); fall back to the available score."""
    ats = _score_value(ats_score)
    semantic = _score_value(semantic_score)
    if ats is not None and semantic is not None:
        return round(0.65 * ats + 0.35 * semantic, 1)
    return ats if ats is not None else semantic


def pipeline_tier(score):
    if score is None:
        return "Not scored"
    return "Strong match" if score >= 70 else "Need Improvement" if score >= 50 else "Low match - Manual Review"


def requirement_breakdown_rows(candidate: dict, company: str, role: str, student_name: str) -> list[dict]:
    rows = []
    for requirement in candidate.get("REQUIREMENT_BREAKDOWN", []) or []:
        checks = requirement.get("checks", []) or []
        assessment = "; ".join(
            f"{check.get('concept', 'Check')}: {check.get('found', 'unknown')}"
            + (f" | {check['evidence']}" if check.get("evidence") else "")
            for check in checks
        )
        rows.append({
            "Company": company,
            "Job Title": role,
            "Student Name (Roster)": student_name,
            "Resume File": candidate.get("file_name", ""),
            "Requirement Type": requirement.get("type", ""),
            "Requirement": requirement.get("requirement", ""),
            "Status": requirement.get("status", ""),
            "Match Score": requirement.get("score", ""),
            "Assessment Details": assessment,
        })
    return rows


def build_excel(
    results: pd.DataFrame,
    company: str,
    role: str,
    offered_ctc: float,
    requirement_breakdown: pd.DataFrame | None = None,
):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        excel_results = results.drop(
            columns=[
                "Identity Status",
                "Identity Confidence",
                "Semantic Score Status",
            ],
            errors="ignore",
        )
        excel_results.to_excel(writer, sheet_name="Candidate Results", index=False)
        if requirement_breakdown is not None:
            requirement_breakdown.to_excel(writer, sheet_name="Requirement Breakdown", index=False)
        summary = pd.DataFrame([
            {"Field": "Company", "Value": company},
            {"Field": "Role / job title", "Value": role},
            {"Field": "Offered CTC (LPA)", "Value": offered_ctc},
            {"Field": "Candidates processed", "Value": len(results)},
            {"Field": "Identity verified", "Value": int((results.get("Identity Status", pd.Series(dtype=str)) == "Verified").sum())},
            {"Field": "CTC eligible", "Value": int((results.get("CTC Eligibility", pd.Series(dtype=str)) == "Eligible").sum())},
            {"Field": "Semantic score available", "Value": int(results.get("Semantic Match Score (%)", pd.Series(dtype=object)).notna().sum())},
        ])
        summary.to_excel(writer, sheet_name="Run Summary", index=False)
        for ws in writer.book.worksheets:
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="234E70")
                cell.alignment = Alignment(wrap_text=True, vertical="top")
            for column_cells in ws.columns:
                max_len = max((len(str(c.value or "")) for c in column_cells[:100]), default=10)
                ws.column_dimensions[get_column_letter(column_cells[0].column)].width = min(max(max_len + 2, 12), 42)
    output.seek(0)
    return output.getvalue()

# ----------------------------- Sidebar ----------------------------------------
with st.sidebar:
    st.header("Connection")
    backend_url = st.text_input("FastAPI base URL", value=os.getenv("ATS_API_URL", "http://localhost:8001"))
    api_key = st.text_input("ATS API key", value=os.getenv("ATS_API_KEY", ""), type="password")
    st.caption("Keep API keys in Streamlit secrets or environment variables in deployment. Do not commit secrets.")

st.subheader("1 · Company and role")
c1, c2 = st.columns(2)
with c1:
    company = st.text_input("Company name", placeholder="e.g. Example Technologies")
    role = st.text_input("Job title", placeholder="e.g. Data Analyst")
with c2:
    offered_ctc = st.number_input("Offered CTC (LPA)", min_value=0.0, max_value=500.0, value=8.0, step=0.5)
    include_breakdown = st.checkbox("Include requirement-level breakdown", value=True)
    ai_feedback = st.checkbox( "Include AI-generated feedback", value=False)
job_description_mode = st.radio(
    "Job description input type",
    ["Text", "PDF", "DOCX"],
    horizontal=True,
)
job_description = ""
if job_description_mode == "Text":
    job_description = st.text_area(
        "Job description",
        height=190,
        placeholder="Paste the complete job description, including required and preferred skills.",
    )
else:
    job_description_file = st.file_uploader(
        f"Upload job description {job_description_mode}",
        type=[job_description_mode.lower()],
    )
    if job_description_file:
        try:
            job_description = extract_text_from_upload(job_description_file).strip()
        except Exception as exc:
            st.error(f"Could not read the job description file: {exc}")
        else:
            if not job_description:
                st.error(
                    "No text could be extracted from the job description file. "
                    "It may contain image-based pages or a viewer-only OCR layer. "
                    "Try exporting it with OCR text embedded, or use DOCX/text input."
                )
            else:
                st.success(f"Extracted {len(job_description)} characters.")
                with st.expander("Preview extracted job description"):
                    st.text(job_description)

st.subheader("2 · Student eligibility data")
st.write("Upload the placement Excel workbook with **Student Name** and one **CTC Range** column (for example, `8–12 LPA`).")
roster_file = st.file_uploader("Placement CTC Excel (.xlsx or .xls)", type=["xlsx", "xls"])
roster_df = None
roster_name_col = roster_range_col = None
if roster_file:
    try:
        roster_df = pd.read_excel(roster_file)
        roster_name_col = get_column(roster_df, ["student name", "name", "student"])
        roster_range_col = get_column(roster_df, ["ctc range", "ctc", "salary range", "eligible ctc range"])
        if roster_name_col is None or roster_range_col is None:
            st.error("Could not identify the required columns. Rename them to 'Student Name' and 'CTC Range'.")
            st.dataframe(roster_df.head(10), use_container_width=True)
            roster_df = None
        else:
            st.success(f"Loaded {len(roster_df)} student records.")
            st.dataframe(roster_df[[roster_name_col, roster_range_col]].head(10), use_container_width=True)
    except Exception as exc:
        st.error(f"Could not read workbook: {exc}")

st.subheader("3 · Resume source")
source_mode = st.radio("Choose resume source", ["Google Drive folder (recommended)", "Upload resumes for a local test"], horizontal=True)
folder_link = ""
uploaded_resumes = []
if source_mode == "Google Drive folder (recommended)":
    folder_link = st.text_input("Google Drive resume folder URL or folder ID", placeholder="https://drive.google.com/drive/folders/...")
    st.caption("The existing FastAPI backend must have access to this folder and support /analyze-folder.")
else:
    uploaded_resumes = st.file_uploader("Upload PDF, DOCX or TXT resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True)
  #  st.caption("Local-upload mode is for UI testing; it does not call the Drive backend's baseline ATS engine.")


st.info("Identity checks are conservative: exact normalized names can be verified; fuzzy matches and duplicate names require manual confirmation. A resume name inferred from its first lines is only a heuristic.")

run_clicked = st.button("Analyze candidates", type="primary", use_container_width=True)
if run_clicked:
    if not company.strip() or not role.strip() or not job_description.strip():
        st.error("Enter the company name, job title and a readable job description.")
        st.stop()
    if not roster_file or roster_df is None:
        st.error("Upload a valid placement CTC Excel workbook first.")
        st.stop()
    if source_mode.startswith("Google Drive") and not folder_link.strip():
        st.error("Enter the Google Drive folder URL.")
        st.stop()
    if not api_key.strip():
        st.error("Enter the FastAPI API key in the sidebar.")
        st.stop()
    if source_mode.startswith("Upload") and not uploaded_resumes:
        st.error("Upload at least one resume.")
        st.stop()

    roster_names = roster_df[roster_name_col].dropna().astype(str).tolist()
    records = []
    requirement_rows = []
    with st.spinner("Analyzing resumes and verifying eligibility..."):
        if source_mode.startswith("Google Drive"):
            try:
                payload = call_backend(
                    backend_url, api_key, folder_link, job_description, include_breakdown,
                    ai_feedback, roster_names,
                )
                candidates = payload.get("candidates", [])
                for candidate in candidates:
                    identity = identity_from_resume_text(
                        "",
                        candidate.get("STUDENT_NAME")
                        or os.path.splitext(candidate.get("file_name", ""))[0],
                        roster_names,
                        candidate.get("STUDENT_NAME_MATCHES"),
                    )
                    matched = identity["matched_name"] if identity["status"] in ("Verified", "Needs review") else ""
                    if include_breakdown:
                        requirement_rows.extend(
                            requirement_breakdown_rows(candidate, company, role, matched)
                        )
                    ctc_range = ""
                    if matched:
                        rows = roster_df[roster_df[roster_name_col].astype(str).map(norm_name) == norm_name(matched)]
                        if len(rows) == 1:
                            ctc_range = rows.iloc[0][roster_range_col]
                    eligibility, eligibility_note = ctc_eligibility(offered_ctc, ctc_range) if matched else ("Needs review", "CTC not evaluated until student identity is resolved")
                    semantic = candidate.get("SEMANTIC_MATCH_SCORE")
                    ats_score = candidate.get("FINAL_ATS_MATCH_SCORE", "")
                    combined_score = calculate_combined_score(ats_score, semantic)
                    if semantic is None:
                        semantic_status = "Backend semantic score not returned; apply backend patch"
                    else:
                        semantic_status = "Available" if semantic is not None else "Disabled"
                    records.append({
                        "Company": company, "Job Title": role, "Offered CTC (LPA)": offered_ctc,
                        "Student Name (Roster)": matched, "Resume File": candidate.get("file_name", ""),
                        "Identity Status": identity["status"], "Identity Confidence": identity["confidence"],
                        "Identity Note": identity["note"], "Student CTC Range": ctc_range,
                        "CTC Eligibility": eligibility, "CTC Eligibility Note": eligibility_note,
                        "Baseline ATS Score (%)": ats_score,
                        "Semantic Match Score (%)": semantic,
                        #"Semantic Score Status": semantic_status,
                        "Combined Score (%)": combined_score,
                        "Pipeline Tier": pipeline_tier(combined_score),
                       # "ATS Routing Tier": candidate.get("PIPELINE_ROUTING_TIER", ""),
                        "Experience (Years)": candidate.get("TOTAL_EXPERIENCE_YEARS", ""),
                        "Matched Criteria": ", ".join(candidate.get("MATCHED_CORE_CRITERIA", [])),
                        "Critical Gaps": ", ".join(candidate.get("CRITICAL_GAPS_IDENTIFIED", [])),
                        "Drive Link": candidate.get("drive_link", ""),
                        "Processing Error": candidate.get("error", ""),
                    })
                for error in payload.get("errors", []):
                    records.append({"Company": company, "Job Title": role, "Offered CTC (LPA)": offered_ctc,
                                    "Resume File": error.get("file_name", ""), "Processing Error": error.get("error", "")})
            except requests.RequestException as exc:
                st.error(f"FastAPI request failed: {exc}")
                st.stop()
            except Exception as exc:
                st.error(f"Could not process backend response: {exc}")
                st.stop()
        else:
            for upload in uploaded_resumes:
                try:
                    # Step 1: Extract resume text in Streamlit
                    text = extract_text_from_upload(upload)

                    if not text or not text.strip():
                        records.append({
                            "Company": company,
                            "Job Title": role,
                            "Offered CTC (LPA)": offered_ctc,
                            "Resume File": upload.name,
                            "Processing Error": "Could not extract text from resume",
                        })
                        continue

                    # Step 2: Run baseline ATS + semantic scoring in main.py
                    candidate = call_text_backend(
                        backend_url=backend_url,
                        api_key=api_key,
                        resume_text=text,
                        job_description=job_description,
                        roster_names=roster_names,
                        ai_feedback=ai_feedback,
                        include_breakdown=include_breakdown,
                    )

                    # Step 3: Verify student identity against the roster
                    identity = identity_from_resume_text(
                        text,
                        candidate.get("STUDENT_NAME", ""),
                        roster_names,
                        candidate.get("STUDENT_NAME_MATCHES"),
                    )

                    matched = (
                        identity["matched_name"]
                        if identity["status"] in ("Verified", "Needs review")
                        else ""
                    )
                    if include_breakdown:
                        requirement_rows.extend(
                            requirement_breakdown_rows(
                                {**candidate, "file_name": upload.name},
                                company,
                                role,
                                matched,
                            )
                        )

                    # Step 4: Check CTC eligibility
                    ctc_range = ""
                    if matched:
                        rows = roster_df[
                            roster_df[roster_name_col].astype(str).map(norm_name)
                            == norm_name(matched)
                        ]
                        if len(rows) == 1:
                            ctc_range = rows.iloc[0][roster_range_col]

                    if matched:
                        eligibility, eligibility_note = ctc_eligibility(
                            offered_ctc, ctc_range
                        )
                    else:
                        eligibility = "Needs review"
                        eligibility_note = (
                            "CTC not evaluated until student identity is resolved"
                        )

                    ats_score = candidate.get("FINAL_ATS_MATCH_SCORE", "")
                    semantic_score = candidate.get("SEMANTIC_MATCH_SCORE")
                    combined_score = calculate_combined_score(ats_score, semantic_score)

                    # Step 5: Save both scoring results
                    records.append({
                        "Company": company,
                        "Job Title": role,
                        "Offered CTC (LPA)": offered_ctc,
                        "Student Name (Roster)": matched,
                        "Resume File": upload.name,
                        "Identity Status": identity["status"],
                        #"Identity Confidence": identity["confidence"],
                        "Identity Note": identity["note"],
                        "Student CTC Range": ctc_range,
                        "CTC Eligibility": eligibility,
                        #"CTC Eligibility Note": eligibility_note,

                        "Baseline ATS Score (%)": ats_score,
                        "Semantic Match Score (%)": semantic_score,
                        #"Semantic Score Status":
                        #    candidate.get("SEMANTIC_SCORE_STATUS", "Not returned"),
                        "Combined Score (%)": combined_score,
                        "Pipeline Tier": pipeline_tier(combined_score),
                        # "ATS Routing Tier":
                        #     candidate.get("PIPELINE_ROUTING_TIER", ""),
                        "Experience of cnadidate (Years)":
                            candidate.get("TOTAL_EXPERIENCE_YEARS", ""),
                        "Matched Criteria": ", ".join(
                            candidate.get("MATCHED_CORE_CRITERIA", [])
                        ),
                        "Critical Gaps": ", ".join(
                            candidate.get("CRITICAL_GAPS_IDENTIFIED", [])
                        ),
                        "AI Feedback": candidate.get("AI_FEEDBACK", "") if ai_feedback else "",
                        "Drive Link": "",
                        "Processing Error": "",
                        
                    })

                except Exception as exc:
                    records.append({
                        "Company": company,
                        "Job Title": role,
                        "Offered CTC (LPA)": offered_ctc,
                        "Resume File": upload.name,
                        "Processing Error": str(exc),
                    })


    results_df = pd.DataFrame(records)
    if results_df.empty:
        st.warning("No candidate results were returned.")
        st.stop()
    # Keep eligible candidates first, then rank by the combined score.
    if "Combined Score (%)" in results_df:
        results_df["_combined_sort"] = pd.to_numeric(results_df["Combined Score (%)"], errors="coerce")
        results_df = results_df.sort_values(
            ["CTC Eligibility", "_combined_sort"], ascending=[True, False], kind="stable"
        ).drop(columns=["_combined_sort"])
    st.session_state["placement_results"] = results_df
    st.session_state["placement_requirement_breakdown"] = requirement_rows
    st.session_state["placement_meta"] = {
        "company": company,
        "role": role,
        "ctc": offered_ctc,
        "include_breakdown": include_breakdown,
    }

if "placement_results" in st.session_state:
    results_df = st.session_state["placement_results"]
    meta = st.session_state.get("placement_meta", {})
    st.divider()
    st.header("Candidate results")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Candidates", len(results_df))
    k2.metric("Identity verified", int((results_df.get("Identity Status", pd.Series(dtype=str)) == "Verified").sum()))
    k3.metric("CTC eligible", int((results_df.get("CTC Eligibility", pd.Series(dtype=str)) == "Eligible").sum()))
    k4.metric("Needs review", int((results_df.get("Identity Status", pd.Series(dtype=str)) == "Needs review").sum()))
    # show_cols = [c for c in [
    #     "Student Name (Roster)", "Resume File", "Identity Status", "Student CTC Range",
    #     "CTC Eligibility", "Baseline ATS Score (%)", "Semantic Match Score (%)",
    #     "ATS Routing Tier", "Identity Note", "CTC Eligibility Note"
    # ] if c in results_df.columns]
    # st.dataframe(results_df[show_cols], use_container_width=True, hide_index=True)
    with st.expander("Full result details"):
        st.dataframe(results_df, use_container_width=True, hide_index=True)
    breakdown_df = None
    if meta.get("include_breakdown", False):
        breakdown_df = pd.DataFrame(
            st.session_state.get("placement_requirement_breakdown", []),
            columns=[
                "Company", "Job Title", "Student Name (Roster)", "Resume File",
                "Requirement Type", "Requirement", "Status", "Match Score",
                "Assessment Details",
            ],
        )
    excel_bytes = build_excel(
        results_df,
        meta.get("company", company),
        meta.get("role", role),
        meta.get("ctc", offered_ctc),
        breakdown_df,
    )
    safe_company = re.sub(r"[^A-Za-z0-9_-]+", "_", meta.get("company", "company")).strip("_") or "company"
    st.download_button(
        "Download Excel report",
        data=excel_bytes,
        file_name=f"placement_ats_{safe_company}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )
    st.caption("The report includes a Candidate Results sheet and a Run Summary sheet. Review identity exceptions and all automated scores before making placement decisions.")
