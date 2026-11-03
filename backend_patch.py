"""
Additions for the existing FastAPI ATS backend.

Integration guide:
1. Add the imports below to the existing backend file.
2. Add the helper functions below.
3. In _score_file(), after resume_text has been extracted and before returning result,
   add the EXTRACTED_STUDENT_NAME and (optionally) SEMANTIC_MATCH_SCORE fields.
4. Keep _run_ats() and FINAL_ATS_MATCH_SCORE unchanged as the baseline engine.
"""

import re
from difflib import SequenceMatcher

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

# Add these lines inside _score_file(), after result = _run_ats(...):
#
# result["EXTRACTED_STUDENT_NAME"] = _extract_student_name(resume_text)
# result.update(_semantic_match_score(resume_text, job_description))
#
# Important: do not change result["FINAL_ATS_MATCH_SCORE"]. It remains the
# deterministic baseline, so both scores can be compared during testing.
