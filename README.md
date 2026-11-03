# Placement ATS v2 — Streamlit UI + FastAPI integration

## What this project includes
- `streamlit_app.py`: Streamlit UI for company name, job title, job description (entered as text or uploaded as PDF/DOCX), offered CTC, roster Excel, and Google Drive folder.
- `backend_patch.py`: minimal additions for the existing FastAPI backend to expose a heuristic extracted name and a separate semantic score.
- `ctc_roster_template.xlsx`: sample workbook with `Student Name` and `CTC Range` columns.
- Excel export with `Candidate Results` and `Run Summary` sheets, plus a `Requirement Breakdown` sheet when the requirement-level breakdown option is enabled.

## Existing backend behavior
The supplied FastAPI backend already:
- extracts PDF/DOCX/TXT/Google Docs resume text from Google Drive;
- calculates `FINAL_ATS_MATCH_SCORE` using its deterministic concept/requirement engine;
- ranks candidates and returns evidence, gaps, experience and routing tiers;
- exposes `/health`, `/analyze`, and `/analyze-folder`;
- optionally calls a local Llama model for resume feedback.

The existing `/analyze-folder` response does not currently expose the extracted resume name or semantic score. Apply `backend_patch.py` so the Streamlit app can display both scoring components. The baseline score is intentionally left unchanged.

## Setup
Use Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 1. Update the FastAPI backend
Copy the helper functions from `backend_patch.py` into the existing backend file. Add the two lines shown in the patch comments inside `_score_file()` after `_run_ats()` returns. Restart FastAPI.

Your backend must already be configured for Google Drive access. For private folders, use a Google service account JSON file and share the Drive folder with that service account as Viewer. Do not make student resumes public just to make the app work.
If the configured service-account file is missing, the backend logs a warning and tries public Drive access; the folder must then be shared as "Anyone with the link can view". Use a valid service-account file path for private folders.

### 2. Run FastAPI
Example (replace `main` with your backend Python filename):
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
Check `http://localhost:8000/health`.

### 3. Run Streamlit
```bash
streamlit run streamlit_app.py
```
In the sidebar, set the FastAPI base URL and API key. Enter the Drive folder link and upload the placement roster workbook.

## Excel roster format
Required columns:
- `Student Name`
- `CTC Range` (one range per student, e.g. `8–12 LPA`)

Accepted range examples: `8-12 LPA`, `8–12 LPA`, `8 to 12 LPA`, `8.5-12 LPA`.
The offered CTC is entered as a numeric value in LPA, e.g. `10.0`.

## Identity safety
For each candidate, the app checks every roster name against the full extracted resume text before falling back to the name heuristic. Exact normalized matches are marked `Verified`; fuzzy matches, duplicate names, missing names, and resumes containing multiple roster names should be manually checked. Name matching is not identity proof; a production version should match a unique student ID or verified institutional email.

## Semantic score caveats
Semantic matching uses `sentence-transformers/all-MiniLM-L6-v2` and cosine similarity displayed as a percentage-like comparative score. It is not a probability of job success and is not calibrated to the baseline ATS score. Compare both on a test set and do not average them until validated. The first model load may download model files and take time.

## Important notes
- Local-upload mode is for UI testing of identity, CTC and semantic matching. It does not run the existing deterministic ATS engine.
- Google Drive mode calls the FastAPI baseline and needs the backend patch to return both the extracted name and semantic score.
- CTC eligibility is inclusive at both endpoints: offered CTC equal to the student's minimum or maximum is eligible.
- Automated ATS scores and eligibility labels should support recruiter review, not automatically decide a student's placement outcome.
- The provided backend has `allow_origins=["*"]` and a default API key fallback of `changeme`. For deployment, restrict CORS to the Streamlit origin, use a strong secret from environment variables, and never deploy with the default key.
- For Streamlit Community Cloud, the FastAPI backend must be reachable over HTTPS, and secrets should be stored in Streamlit Secrets rather than source code.
