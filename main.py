import html
import io
import json
import logging
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import requests
from docx import Document
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from google.auth.transport.requests import AuthorizedSession
from google.oauth2 import service_account
from llama_cpp import Llama
from pydantic import BaseModel, Field
from pypdf import PdfReader

logger = logging.getLogger(__name__)

load_dotenv()

API_KEY    = os.getenv("ATS_API_KEY", "changeme")
MODEL_REPO = os.getenv("MODEL_REPO", "bartowski/Llama-3.2-3B-Instruct-GGUF")
MODEL_FILE = os.getenv("MODEL_FILE", "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

app = FastAPI(title="ATS Resume Analyzer", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_llm: Llama | None = None


def _ensure_model() -> str:
    local_path = os.path.join(MODELS_DIR, MODEL_FILE)
    if os.path.exists(local_path):
        return local_path
    print(f"Downloading {MODEL_FILE} from HuggingFace...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=MODELS_DIR)


def get_llm() -> Llama:
    global _llm
    if _llm is None:
        model_path = _ensure_model()
        _llm = Llama(model_path=model_path, n_ctx=4096, n_threads=os.cpu_count() or 4, verbose=False)
    return _llm


# ---------------------------------------------------------------------------
# Deterministic ATS engine
# ---------------------------------------------------------------------------

# (label, what the JD says, what counts as evidence in the resume)
CONCEPTS = [
    ("programming languages", r"programming languages?|modern languages?|coding",
     r"\b(java|python|javascript|typescript|c\+\+|c#|golang|kotlin|scala|ruby|rust|php)\b"),
    ("databases / SQL", r"database|\bsql\b|query languages?",
     r"\b(sql|mysql|postgres\w*|oracle|mongodb|nosql|redis|dynamodb)\b"),
    ("system design / architecture", r"system design|architect|design artifacts|scalable",
     r"system design|architect|scalab|microservices|high-throughput|distributed"),
    ("application development", r"application development|software solutions|production code|develop",
     r"spring|fastapi|django|flask|node|react|developed|built|implemented|designed"),
    ("testing", r"\btest", r"\btest|junit|pytest|mockito|selenium|postman|\bqa\b"),
    ("operational stability / resiliency", r"operational stability|resilien|stable|reliab",
     r"uptime|reliab|resilien|monitor|\d{2}\.\d+%|high.availability|fault.toleran"),
    ("debugging / troubleshooting", r"debug|troubleshoot|hidden problems|coding hygiene",
     r"debug|troubleshoot|optimi[sz]|latency|bottleneck|refactor"),
    ("large corporate / enterprise environment", r"corporate|enterprise(?!-authorized)",
     r"enterprise|j\.?\s?p\.?\s?morgan|capgemini|bank|insurance|fintech|financial"),
    ("security", r"secur", r"secur|jwt|rbac|oauth|encrypt|cryptograph|compliance|compliant"),
    ("data sensitivity", r"data sensitivity|sensitive", r"sensitive|compliance|compliant|\bpii\b"),
    ("CI/CD", r"ci/cd|continuous (integration|delivery|deployment)",
     r"ci/cd|github actions|jenkins|gitlab|circleci|continuous (deploy|integrat)"),
    ("agile", r"agile|scrum", r"agile|scrum|kanban|sprint"),
    ("SDLC", r"\bsdlc\b|software development life ?cycle",
     r"\bsdlc\b|agile|scrum|ci/cd|github actions|jira|deploy"),
    ("AI-assisted development", r"\bai\b|artificial intelligence|\bllm|gen ?ai|copilot",
     r"copilot|chatgpt|claude|cursor|codeium|tabnine|gen ?ai|\bllm|ai-assisted|ai assisted|prompt engineering"),
    ("frontend", r"front-?end|\bui\b|user interface",
     r"react|angular|vue|javascript|typescript|html|css|\bui\b"),
    ("cloud", r"cloud|\baws\b|azure|\bgcp\b",
     r"\baws\b|azure|\bgcp\b|google cloud|\bec2\b|\bs3\b|lambda|kubernetes"),
    ("data analysis / visualization", r"datasets?|visuali[sz]|reports?\b|analy[sz]",
     r"dashboard|report|analytics|visuali[sz]|echarts|tableau|power bi|ag grid"),
    ("automation", r"automat", r"automat|scheduler|github actions|pipeline"),
    ("collaboration / mentoring", r"share learnings|guide peers|mentor|team effectiveness|collaborat",
     r"collaborat|mentor|\bled\b|guided|cross-functional|partnered"),
    ("formal training", r"formal training|degree|bachelor|certification",
     r"b\.?\s?tech|b\.?\s?e\.|bachelor|master|m\.?\s?tech|degree|certif"),
]
CONCEPTS = [(label, re.compile(jd, re.I), re.compile(res, re.I)) for label, jd, res in CONCEPTS]
# Broad concepts that appear in most bullets; half weight so a bullet's specific subject decides it
_GENERIC = {"programming languages", "application development", "testing", "debugging / troubleshooting",
            "automation", "collaboration / mentoring"}

_TYPE_WEIGHT = {"required": 2.0, "responsibility": 1.0, "preferred": 1.0}
_KNOCKOUT_TERMS = re.compile(
    r"[^.\n]*(authori[sz]ation|visa|sponsor|citizen|clearance|relocat|must (?:be|reside|live)"
    r"|on-?site only|licen[sc]e)[^.\n]*",
    re.I,
)
_YEARS = re.compile(r"(\d+)\+?\s*(?:or more\s+)?years?", re.I)
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
_DATE_RANGE = re.compile(
    r"([A-Za-z]{3})[a-z]*\.?\s+(\d{4})\s*(?:-|to)\s*(?:([A-Za-z]{3})[a-z]*\.?\s+(\d{4})|(present|current|now))",
    re.I,
)
_BULLET = re.compile(r"^\s*(?:[-*•▪●]|\d+[.)])\s+(.+)")
_STOPWORDS = set(
    "the and for with that this from your you our are will have has into across using use able "
    "ability their within what such other more than least well also both".split()
)


def _normalize(text: str) -> str:
    """ASCII-fy dashes/bullets/quotes and collapse whitespace."""
    for src, dst in {"–": "-", "—": "-", "•": "-", "�": "-", "‘": "'", "’": "'", "“": '"', "”": '"'}.items():
        text = text.replace(src, dst)
    return re.sub(r"[ \t]+", " ", text)


def _sections(resume: str) -> dict[str, str]:
    """Split resume on ALL-CAPS heading lines -> {heading: body}."""
    out, current = {"_top": []}, "_top"
    for line in resume.splitlines():
        s = line.strip()
        if re.fullmatch(r"[A-Z][A-Z &/]{3,}", s):
            current = s
            out.setdefault(current, [])
        else:
            out[current].append(line)
    return {k: "\n".join(v) for k, v in out.items()}


def _experience_years(experience_text: str) -> float:
    """Total months covered by date ranges (overlaps merged), in years."""
    today = date.today()
    spans = []
    for sm, sy, em, ey, present in _DATE_RANGE.findall(experience_text):
        start_m = _MONTHS.get(sm.lower())
        if not start_m:
            continue
        start = int(sy) * 12 + start_m
        if present:
            end = today.year * 12 + today.month
        elif _MONTHS.get(em.lower()):
            end = int(ey) * 12 + _MONTHS[em.lower()]
        else:
            continue
        if end > start:
            spans.append((start, end))
    months, cur_end = 0, 0
    for s, e in sorted(spans):
        s = max(s, cur_end)
        if e > s:
            months += e - s
            cur_end = e
    return round(months / 12, 1)


def _jd_requirements(jd: str) -> list[tuple[str, str]]:
    """Return [(type, requirement_text)] from JD bullet points, typed by the section they sit in."""
    reqs, kind = [], "responsibility"
    for line in jd.splitlines():
        m = _BULLET.match(line)
        if m:
            reqs.append((kind, m.group(1).strip()))
            continue
        low = line.lower()
        if re.search(r"preferred|nice to have|bonus|plus\b", low):
            kind = "preferred"
        elif re.search(r"required|requirements|qualifications|must have|skills|who you are", low):
            kind = "required"
        elif re.search(r"responsibilit|what you.ll do|duties|role", low):
            kind = "responsibility"
    if not reqs:  # unstructured JD: treat each sentence as a requirement
        reqs = [("required", s.strip()) for s in re.split(r"(?<=[.;])\s+", jd) if len(s.strip()) > 25]
    return reqs


def _evidence(pattern: re.Pattern, text: str) -> str:
    for line in text.splitlines():
        m = pattern.search(line)
        if m:
            line = line.strip(" -")
            start = max(0, m.start() - 40)
            return ("..." if start else "") + line[start:start + 110].strip()
    return ""


def _evaluate(kind: str, req: str, context: str, full: str, years: float) -> dict:
    """Score one requirement. Skills seen only in a skills list (no job/project context) get half credit."""
    checks = []
    m = _YEARS.search(req)
    if m:
        need = int(m.group(1))
        pts = 1.0 if years >= need else 0.5 if years >= need * 0.6 else 0.0
        checks.append((f"{need}+ years experience", pts, f"{years} years computed from resume dates"))
    for label, jd_pat, res_pat in CONCEPTS:
        if not jd_pat.search(req):
            continue
        ev = _evidence(res_pat, context)
        if ev:
            checks.append((label, 1.0, ev))
        elif res_pat.search(full):
            checks.append((label, 0.5, "listed in skills only, no job/project context"))
        else:
            checks.append((label, 0.0, ""))

    if checks:
        weights = [0.5 if c[0] in _GENERIC else 1.0 for c in checks]
        points = sum(w * c[1] for w, c in zip(weights, checks)) / sum(weights)
    else:  # no known concept: fall back to content-word overlap
        words = {w for w in re.findall(r"[a-z][a-z+#/-]{3,}", req.lower()) if w not in _STOPWORDS}
        hits = [w for w in words if w in full.lower()]
        points = min(1.0, len(hits) / len(words) * 1.5) if words else 0.0
        checks = [("keyword overlap", points, ", ".join(sorted(hits)[:6]))]

    status = "met" if points >= 0.75 else "partial" if points >= 0.25 else "missing"
    return {
        "requirement": req if len(req) <= 140 else req[:137] + "...",
        "type": kind,
        "status": status,
        "score": round(points, 2),
        "checks": [{"concept": label, "found": "yes" if pts == 1.0 else "partial" if pts else "no",
                    "evidence": ev} for label, pts, ev in checks],
    }


def _run_ats(resume_text: str, job_description: str) -> dict:
    resume = _normalize(resume_text)
    jd     = _normalize(job_description)
    secs   = _sections(resume)
    context = "\n".join(v for k, v in secs.items() if "SKILL" not in k)  # everything except skill lists
    exp_text = "\n".join(v for k, v in secs.items() if "EXPERIENCE" in k) or resume
    years = _experience_years(exp_text)

    results = [_evaluate(kind, req, context, resume, years) for kind, req in _jd_requirements(jd)]
    total  = sum(_TYPE_WEIGHT[r["type"]] for r in results)
    earned = sum(_TYPE_WEIGHT[r["type"]] * r["score"] for r in results)
    score  = round(100 * earned / total) if total else 0

    knockout_hits = [m.group(0).strip(" -") for m in _KNOCKOUT_TERMS.finditer(jd)]
    if knockout_hits:
        knockout = "Review - JD states hard requirement(s): " + "; ".join(knockout_hits[:3])
    else:
        knockout = "Pass - no hard dealbreakers stated in the job description"

    tier = "Strong match" if score >= 80 else "Need Improvement" if score >= 50 else "Low match - Manual Review"

    # Gaps: missing concepts, required ones first
    gap_order = sorted(results, key=lambda r: (r["type"] != "required", r["score"]))
    gaps, seen = [], set()
    for r in gap_order:
        for c in r["checks"]:
            if c["found"] != "yes":
                c["concept"] = (
                r["requirement"]
                if c["concept"] == "keyword overlap" else c["concept"])
                if  c["concept"] not in seen :
                    seen.add(c["concept"])
                    gaps.append((c["concept"], c["found"], r["type"]))

    return {
        "KNOCKOUT_STATUS": knockout,
        "FINAL_ATS_MATCH_SCORE": f"{score}%",
        "PIPELINE_ROUTING_TIER": tier,
        "TOTAL_EXPERIENCE_YEARS": years,
        "MATCHED_CORE_CRITERIA": sorted(
            {c["concept"] for r in results for c in r["checks"]
             if c["found"] == "yes" and c["concept"] != "keyword overlap"}
        ),
        "CRITICAL_GAPS_IDENTIFIED": [
            f"{g} ({t}{', only in skills list' if f == 'partial' else ''})" for g, f, t in gaps[:5]
        ],
        "OPTIMIZATION_RECOMMENDATIONS": [
            (f"Show '{g}' in a job or project bullet (what you did, tools, measurable result)"
             if f == "partial" else
             f"Add concrete experience with '{g}' to the resume if you have it (tool, where used, outcome)")
            for g, f, _ in gaps[:3]
        ],
        "REQUIREMENT_BREAKDOWN": results,
    }


def _ai_feedback(resume_text: str, job_description: str, ats: dict) -> list[str]:
    """Optional: ask the local LLM for 3 short tailored tips (slow on CPU, ~30-60s)."""
    prompt = (
        f"Job title/summary: {_normalize(job_description)[:600]}\n"
        f"Candidate gaps: {', '.join(ats['CRITICAL_GAPS_IDENTIFIED'])}\n"
        f"Candidate strengths: {', '.join(ats['MATCHED_CORE_CRITERIA'])}\n\n"
        "Give exactly 3 short, specific resume improvement tips as a JSON array of strings."
    )
    try:
        out = get_llm().create_chat_completion(
            messages=[{"role": "user", "content": prompt}], max_tokens=250, temperature=0.2,
        )
        raw = out["choices"][0]["message"]["content"]
        m = re.search(r"\[[\s\S]*\]", raw)
        return json.loads(m.group(0))[:3] if m else [raw.strip()]
    except Exception as e:
        return [f"AI feedback unavailable: {e}"]


# ---------------------------------------------------------------------------
# Resume download / extraction
# ---------------------------------------------------------------------------

def require_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _extract_file_id(drive_link: str) -> str:
    for pattern in [r"/file/d/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)", r"/d/([a-zA-Z0-9_-]+)"]:
        m = re.search(pattern, drive_link)
        if m:
            return m.group(1)
    raise ValueError(f"Cannot parse file ID from: {drive_link}")


GOOGLE_SA_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "")
DRIVE_API = "https://www.googleapis.com/drive/v3/files"
_drive: AuthorizedSession | None = None


def _drive_session() -> AuthorizedSession | None:
    """Authenticated Drive session from the service-account JSON, or None if not configured."""
    global _drive
    if _drive is None and GOOGLE_SA_FILE:
        if not os.path.isfile(GOOGLE_SA_FILE):
            logger.warning(
                "Configured Google service-account file is missing; "
                "trying public Google Drive access instead."
            )
            return None
        creds = service_account.Credentials.from_service_account_file(
            GOOGLE_SA_FILE, scopes=["https://www.googleapis.com/auth/drive.readonly"]
        )
        _drive = AuthorizedSession(creds)
    return _drive


def _download_api(file_id: str) -> tuple[bytes, str]:
    """Download via Drive API; native Google Docs are exported as plain text."""
    s = _drive_session()
    r = s.get(f"{DRIVE_API}/{file_id}", params={"alt": "media", "supportsAllDrives": "true"}, timeout=60)
    if r.status_code == 403 and "fileNotDownloadable" in r.text:
        r = s.get(f"{DRIVE_API}/{file_id}/export", params={"mimeType": "text/plain"}, timeout=60)
    r.raise_for_status()
    return r.content, r.headers.get("Content-Type", "")


def _download(file_id: str) -> tuple[bytes, str]:
    base_url = "https://drive.google.com/uc"
    s = requests.Session()
    r = s.get(base_url, params={"export": "download", "id": file_id}, timeout=30)
    r.raise_for_status()
    confirm = next((v for k, v in r.cookies.items() if "warning" in k), None)
    if confirm:
        r = s.get(base_url, params={"export": "download", "id": file_id, "confirm": confirm}, timeout=60)
        r.raise_for_status()
    return r.content, r.headers.get("Content-Type", "")


def _extract_text(content: bytes, content_type: str) -> str:
    if "pdf" in content_type or content[:4] == b"%PDF":
        reader = PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if "word" in content_type or "docx" in content_type or content[:2] == b"PK":
        doc = Document(io.BytesIO(content))
        return "\n".join(p.text for p in doc.paragraphs)
    return content.decode("utf-8", errors="ignore")


def _download_gdoc(file_id: str) -> tuple[bytes, str]:
    """Native Google Docs can't be downloaded directly; export them as plain text."""
    r = requests.get(f"https://docs.google.com/document/d/{file_id}/export", params={"format": "txt"}, timeout=30)
    r.raise_for_status()
    return r.content, "text/plain"
''' Name search for CTC'''
def _extract_student_name(resume_text: str) -> str:
    """
    Conservative heuristic. Resume layout varies, so this must not be treated as
    authoritative identity proof. Prefer matching a unique student email/ID where
    available in a future version.
    """
    ignored = re.compile(
        r"@|linkedin|github|resume|curriculum vitae|phone|mobile|address|portfolio|"
        r"computer engineering|data analyst|software engineer|objective|summary",
        re.I,
    )
    for line in resume_text.splitlines()[:12]:
        line = re.sub(r"\s+", " ", line).strip(" |•-\t")
        if not line or len(line) > 70 or ignored.search(line):
            continue
        words = re.findall(r"[A-Za-z][A-Za-z.'-]*", line)
        if 2 <= len(words) <= 5 and len(words) == len(line.split()):
            return line
    return ""

def _normalize_person_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9 ]", " ", value.lower()).strip()


def _find_roster_name_matches(resume_text: str, roster_names: list[str]) -> list[str]:
    normalized_text = f" {_normalize_person_name(resume_text)} "
    matches = []
    seen = set()
    for name in roster_names:
        normalized_name = _normalize_person_name(name)
        if normalized_name and normalized_name not in seen and f" {normalized_name} " in normalized_text:
            matches.append(name)
            seen.add(normalized_name)
    return matches

def _semantic_match_score(resume_text: str, job_description: str) -> dict:
    """
    Optional second scoring component. Does not replace the deterministic score.
    Install sentence-transformers and scikit-learn to enable.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity

        global _semantic_model
        try:
            _semantic_model
        except NameError:
            _semantic_model = None
        if _semantic_model is None:
            model_name = os.getenv(
                "SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            )
            _semantic_model = SentenceTransformer(model_name)
        vectors = _semantic_model.encode(
            [resume_text[:12000], job_description[:12000]],
            normalize_embeddings=True,
        )
        cosine = float(cosine_similarity([vectors[0]], [vectors[1]])[0][0])
        score = round(max(0.0, min(100.0, cosine * 100)), 1)
        return {"SEMANTIC_MATCH_SCORE": score, "SEMANTIC_SCORE_STATUS": "Available"}
    except Exception as exc:
        return {
            "SEMANTIC_MATCH_SCORE": None,
            "SEMANTIC_SCORE_STATUS": f"Unavailable: {type(exc).__name__}",
        }
'''adding semantic layer'''

def _score_file(
    file_id: str,
    job_description: str,
    ai_feedback: bool,
    is_gdoc: bool = False,
    roster_names: list[str] | None = None,
) -> dict:
    try:
        try:
            if not _drive_session():
                raise LookupError
            content, content_type = _download_api(file_id)
        except (LookupError, requests.HTTPError):
            # No service account, or file not shared with it: try the public "anyone with link" route
            content, content_type = _download_gdoc(file_id) if is_gdoc else _download(file_id)
    except Exception as e:
        return {"error": f"Download failed: {e}"}
    try:
        resume_text = _extract_text(content, content_type)
    except Exception as e:
        return {"error": f"Could not read file: {e}"}
    if not resume_text.strip():
        return {"error": "Could not extract text from file"}
    result = _run_ats(resume_text, job_description)
    roster_matches = _find_roster_name_matches(resume_text, roster_names or [])
    result["STUDENT_NAME_MATCHES"] = roster_matches
    result["STUDENT_NAME"] = (
        roster_matches[0] if len(roster_matches) == 1 else _extract_student_name(resume_text)
    )
    result.update(_semantic_match_score(resume_text, job_description))
    if ai_feedback:
        result["AI_FEEDBACK"] = _ai_feedback(resume_text, job_description, result)
    return result


def _process_one(drive_link: str, job_description: str, ai_feedback: bool) -> dict:
    try:
        file_id = _extract_file_id(drive_link)
    except ValueError as e:
        return {"drive_link": drive_link, "error": str(e)}
    result = _score_file(file_id, job_description, ai_feedback, is_gdoc="docs.google.com/document" in drive_link)
    result["drive_link"] = drive_link
    return result


# ---------------------------------------------------------------------------
# Google Drive folder listing
# ---------------------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
_RESUME_EXT = (".pdf", ".docx", ".txt")
_FOLDER_MIME = "application/vnd.google-apps.folder"
_GDOC_MIME = "application/vnd.google-apps.document"


def _extract_folder_id(folder_link: str) -> str:
    for pattern in [r"/folders/([a-zA-Z0-9_-]+)", r"id=([a-zA-Z0-9_-]+)"]:
        m = re.search(pattern, folder_link)
        if m:
            return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{10,}", folder_link.strip()):
        return folder_link.strip()
    raise ValueError(f"Cannot parse folder ID from: {folder_link}")


def _list_folder_api(folder_id: str) -> list[dict]:
    """Official Drive API v3 listing via service account (or GOOGLE_API_KEY for public folders)."""
    session = _drive_session()
    if session:  # Drive returns an empty list (not an error) for unshared folders, so check access first
        r = session.get(f"{DRIVE_API}/{folder_id}", params={"fields": "id", "supportsAllDrives": "true"}, timeout=30)
        if r.status_code == 404:
            raise ValueError(
                f"Folder not accessible. Share it (Viewer) with {session.credentials.service_account_email}"
            )
    items, token = [], None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed = false",
            "fields": "nextPageToken, files(id, name, mimeType)",
            "pageSize": 1000, "pageToken": token,
            "supportsAllDrives": "true", "includeItemsFromAllDrives": "true",
        }
        if session:
            r = session.get(DRIVE_API, params=params, timeout=30)
        else:
            r = requests.get(DRIVE_API, params={**params, "key": GOOGLE_API_KEY}, timeout=30)
        r.raise_for_status()
        data = r.json()
        for f in data.get("files", []):
            kind = "folder" if f["mimeType"] == _FOLDER_MIME else "gdoc" if f["mimeType"] == _GDOC_MIME else "file"
            items.append({"id": f["id"], "name": f["name"], "kind": kind})
        token = data.get("nextPageToken")
        if not token:
            return items


def _list_folder_public(folder_id: str) -> list[dict]:
    """No-key listing by reading Drive's public embedded folder view."""
    r = requests.get("https://drive.google.com/embeddedfolderview", params={"id": folder_id}, timeout=30)
    if r.status_code == 404:
        raise ValueError("Folder not found or not shared as 'Anyone with the link can view'")
    r.raise_for_status()
    items = []
    for entry in re.finditer(r'<div class="flip-entry" id="entry-([\w-]+)"(.*?)</a>', r.text, re.S):
        file_id, body = entry.groups()
        href = re.search(r'href="([^"]+)"', body)
        title = re.search(r'class="flip-entry-title">(.*?)</div>', body, re.S)
        href = href.group(1) if href else ""
        kind = "folder" if "/folders/" in href else "gdoc" if "docs.google.com/document" in href else "file"
        items.append({"id": file_id, "name": html.unescape(title.group(1).strip()) if title else file_id, "kind": kind})
    return items


def _list_resumes(folder_id: str, recursive: bool, depth: int = 0) -> list[dict]:
    use_api = bool(GOOGLE_API_KEY or (GOOGLE_SA_FILE and os.path.isfile(GOOGLE_SA_FILE)))
    items = _list_folder_api(folder_id) if use_api else _list_folder_public(folder_id)
    resumes = []
    for it in items:
        if it["kind"] == "folder":
            if recursive and depth < 3:
                resumes += _list_resumes(it["id"], recursive, depth + 1)
        elif it["kind"] == "gdoc" or it["name"].lower().endswith(_RESUME_EXT):
            resumes.append(it)
    return resumes


class AnalyzeRequest(BaseModel):
    drive_links: list[str] = Field(
        ...,
        description="One or more Google Drive shareable links to resumes (PDF or DOCX)",
        min_length=1,
    )
    job_description: str = Field(..., description="Full job description text")
    ai_feedback: bool = Field(
        False, description="Also ask the local LLM for 3 tailored tips (adds ~30-60s per resume on CPU)"
    )


@app.get("/health")
def health():
    model_path = os.path.join(MODELS_DIR, MODEL_FILE)
    return {
        "status": "ok",
        "model": MODEL_FILE,
        "model_downloaded": os.path.exists(model_path),
        "model_loaded": _llm is not None,
    }


@app.post("/analyze", dependencies=[Depends(require_api_key)])
def analyze(request: AnalyzeRequest):
    """
    Score one or more resumes (Google Drive links) against a job description.
    Header: X-API-Key  |  Returns a list — one result per drive link.
    """
    return [_process_one(link, request.job_description, request.ai_feedback) for link in request.drive_links]


class AnalyzeFolderRequest(BaseModel):
    folder_link: str = Field(..., description="Google Drive folder link (shared as 'Anyone with the link can view')")
    job_description: str = Field(..., description="Full job description text")
    roster_names: list[str] = Field(default_factory=list, description="Roster names to find anywhere in each resume")
    include_subfolders: bool = Field(True, description="Also scan resumes in subfolders (up to 3 levels)")
    include_breakdown: bool = Field(False, description="Include the per-requirement breakdown for each resume")
    ai_feedback: bool = Field(
        False, description="Also ask the local LLM for 3 tailored tips (adds ~30-60s per resume on CPU)"
    )


@app.post("/analyze-folder", dependencies=[Depends(require_api_key)])
def analyze_folder(request: AnalyzeFolderRequest):
    """
    Score every resume (PDF / DOCX / TXT / Google Doc) in a Google Drive folder against a job description.
    Header: X-API-Key  |  Returns candidates ranked by ATS score, highest first.
    """
    try:
        folder_id = _extract_folder_id(request.folder_link)
        resumes = _list_resumes(folder_id, request.include_subfolders)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Could not list Drive folder: {e}")
    if not resumes:
        raise HTTPException(status_code=404, detail="No PDF/DOCX/TXT/Google Doc resumes found in the folder")

    def work(item: dict) -> dict:
        result = _score_file(
            item["id"], request.job_description, request.ai_feedback,
            item["kind"] == "gdoc", request.roster_names,
        )
        if not request.include_breakdown:
            result.pop("REQUIREMENT_BREAKDOWN", None)
        return {"file_name": item["name"], "drive_link": f"https://drive.google.com/file/d/{item['id']}/view",
                **result}

    # Downloads are I/O-bound -> parallel; LLM feedback is CPU-bound -> sequential
    with ThreadPoolExecutor(max_workers=1 if request.ai_feedback else 8) as pool:
        results = list(pool.map(work, resumes))

    scored = sorted((r for r in results if "error" not in r),
                    key=lambda r: int(r["FINAL_ATS_MATCH_SCORE"].rstrip("%")), reverse=True)
    for rank, r in enumerate(scored, 1):
        r["rank"] = rank
    tiers = [r["PIPELINE_ROUTING_TIER"] for r in scored]
    return {
        "folder_id": folder_id,
        "total_resumes": len(results),
        "scored": len(scored),
        "failed": len(results) - len(scored),
        "tier_summary": {t: tiers.count(t) for t in ("Auto-Advance", "Manual Review", "Delayed Archive")},
        "candidates": scored,
        "errors": [r for r in results if "error" in r],
    }



class AnalyzeTextRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)
    job_description: str = Field(..., min_length=1)
    roster_names: list[str] = Field(default_factory=list)
    ai_feedback: bool = False


@app.post("/analyze-text", dependencies=[Depends(require_api_key)])
def analyze_text(request: AnalyzeTextRequest):
    """Score one locally uploaded resume using existing ATS and semantic logic."""

    resume_text = request.resume_text.strip()
    job_description = request.job_description.strip()

    if not resume_text:
        raise HTTPException(status_code=400, detail="Resume text is empty")

    result = _run_ats(resume_text, job_description)
    roster_matches = _find_roster_name_matches(resume_text, request.roster_names)
    result["STUDENT_NAME_MATCHES"] = roster_matches
    result["STUDENT_NAME"] = (
        roster_matches[0] if len(roster_matches) == 1 else _extract_student_name(resume_text)
    )
    result.update(_semantic_match_score(resume_text, job_description))

    if request.ai_feedback:
        result["AI_FEEDBACK"] = _ai_feedback(
            resume_text, job_description, result
        )

    return result
