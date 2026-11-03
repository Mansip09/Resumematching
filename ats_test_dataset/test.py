import argparse, csv, json, os, re
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
RESUMES = ROOT / "resumes"
JDS = ROOT / "job_descriptions"
META = ROOT / "dataset_metadata.json"
EXPECTED = DATA / "expected_scores.csv"

def norm(s):
    return re.sub(r"[^a-z0-9+#.]+", " ", str(s).lower()).strip()

def score_pair(resume, jd):
    """Reference rubric: required skills 60, preferred skills 15, experience 15, domain 10."""
    rs = [norm(s) for s in resume["skills"]]
    req = [norm(s) for s in jd["required"]]
    pref = [norm(s) for s in jd["preferred"]]
    req_cov = sum(any(s == x or s in x or x in s for x in rs) for s in req) / max(1, len(req))
    pref_cov = sum(any(s == x or s in x or x in s for x in rs) for s in pref) / max(1, len(pref))
    exp_fit = min(max(0, int(resume["years_experience"])) / max(1, int(jd["min_years"])), 1.0)
    domain = 1.0 if norm(resume["domain"]) == norm(jd["domain"]) else 0.0
    score = round(60*req_cov + 15*pref_cov + 15*exp_fit + 10*domain)
    return max(0, min(100, score)), req_cov, pref_cov, exp_fit, bool(domain)

def load_meta():
    data = json.loads(META.read_text(encoding="utf-8"))
    return data["resumes"], data["job_descriptions"]

def generate_expected():
    DATA.mkdir(exist_ok=True)
    resumes, jobs = load_meta()
    rows = []
    for r in resumes:
        for j in jobs:
            score, req, pref, exp, domain = score_pair(r,j)
            rows.append({
                "resume_id":r["id"], "resume_name":r["name"], "resume_file":r["file"],
                "jd_id":j["id"], "jd_title":j["title"], "jd_file":j["file"],
                "expected_score":score,
            
                "required_skill_coverage_pct":round(req*100,1),
                "preferred_skill_coverage_pct":round(pref*100,1),
                "experience_fit_pct":round(exp*100,1), "domain_match":domain
            })
    with EXPECTED.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"Created {len(rows)} reference scores at {EXPECTED}")
    return rows

def call_api(base_url, api_key, resume_text, jd_text, ai_feedback=False):
    # Update this header if your main.py expects a different API-key header.
    r = requests.post(base_url.rstrip("/")+"/analyze-text",
        headers={"X-API-Key":api_key},
        json={"resume_text":resume_text,"job_description":jd_text,"ai_feedback":ai_feedback},
        timeout=180)
    r.raise_for_status()
    return r.json()

def get_score(payload, keys):
    for k in keys:
        try:
            if payload.get(k) not in (None, ""):
                return float(payload[k])
        except (TypeError, ValueError):
            pass
    return None

def run_api(base_url, api_key, ai_feedback=False):
    resumes, jobs = load_meta()
    expected = {(x["resume_id"],x["jd_id"]):float(x["expected_score"]) for x in generate_expected()}
    results=[]
    total=len(resumes)*len(jobs)
    for i,r in enumerate(resumes):
        rt=(RESUMES/r["file"]).read_text(encoding="utf-8")
        for j in jobs:
            jt=(JDS/j["file"]).read_text(encoding="utf-8")
            row={"resume_id":r["id"],"resume_name":r["name"],"jd_id":j["id"],"jd_title":j["title"]}
            try:
                payload=call_api(base_url,api_key,rt,jt,ai_feedback)
                actual=get_score(payload,["FINAL_ATS_MATCH_SCORE","ATS_MATCH_SCORE","FINAL_SCORE"])
                semantic=get_score(payload,["SEMANTIC_MATCH_SCORE","SEMANTIC_SCORE"])
                row.update({"actual_ats_score":actual,"semantic_match_score":semantic,
                    "semantic_score_status":payload.get("SEMANTIC_SCORE_STATUS",""),
                    "ai_feedback_present":bool(payload.get("AI_FEEDBACK")),
                    "api_status":"OK" if actual is not None else "MISSING_SCORE",
                    "error":"" if actual is not None else f"No recognized ATS score field. Keys: {list(payload)}"})
            except Exception as e:
                row.update({"actual_ats_score":"","semantic_match_score":"","semantic_score_status":"",
                    "ai_feedback_present":False,"api_status":"ERROR","error":str(e)})
            exp=expected[(r["id"],j["id"])]
            try: err=abs(float(row["actual_ats_score"])-exp)
            except (ValueError,TypeError): err=""
            row["rubric_expected_score"]=exp
            row["absolute_error_vs_rubric"]=round(err,2) if err!="" else ""
            results.append(row)
        print(f"Completed resume {i+1}/{len(resumes)} ({(i+1)*len(jobs)} of {total} pairs)")
    out=DATA/"score_comparison.csv"
    with out.open("w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
    ok=[x for x in results if x["actual_ats_score"] not in ("",None)]
    errs=[x["absolute_error_vs_rubric"] for x in ok if x["absolute_error_vs_rubric"]!=""]
    print(f"\nSaved API comparisons: {out}")
    print(f"Returned ATS score for {len(ok)}/{total} combinations")
    if errs: print(f"Mean absolute difference from synthetic rubric: {sum(errs)/len(errs):.2f} points")
    print("This difference measures agreement with this simple rubric, not real-world hiring accuracy.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--generate-only",action="store_true",help="Only generate the 100 reference scores.")
    p.add_argument("--api-url",default="http://127.0.0.1:8001")
    p.add_argument("--api-key",default=os.getenv("ATS_API_KEY",""))
    p.add_argument("--ai-feedback",action="store_true",help="Enable AI feedback on all 100 requests; may take longer.")
    a=p.parse_args()
    generate_expected()
    if a.generate_only:
        print("No API calls made.")
        return
    if not a.api_key:
        raise SystemExit("Missing API key. Pass --api-key YOUR_KEY or set ATS_API_KEY.")
    run_api(a.api_url,a.api_key,a.ai_feedback)

if __name__=="__main__":
    main()
