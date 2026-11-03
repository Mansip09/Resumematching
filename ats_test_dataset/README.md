# Synthetic ATS validation dataset

Contains 10 fictional resumes, 10 fictional job descriptions and 100 reference scores (every resume paired with every JD).

## Scoring rubric (0-100)
- 60 points: coverage of required skills
- 15 points: coverage of preferred skills
- 15 points: experience fit, proportional up to the JD minimum
- 10 points: domain match

`expected_scores.csv` includes the 100 reference scores and rubric components. These are transparent synthetic rubric scores, not expert-labelled hiring ground truth and not the output of your current ATS. Use them to check broad ranking behavior and score differences, not to claim proven real-world accuracy.

## Run
1. Extract this zip.
2. In the extracted folder, install dependency if needed: `pip install requests`
3. Start your FastAPI app in another terminal (example): `uvicorn main:app --reload --port 8001`
4. Generate only the expected scores: `python test.py --generate-only`
5. Run all 100 pairs against the API: `python test.py --api-url http://127.0.0.1:8001 --api-key YOUR_API_KEY`
6. Optional feedback test: append `--ai-feedback` (slower, depending on `_ai_feedback` implementation).

The API key header in `test.py` is `X-API-Key`. If `require_api_key` in your `main.py` expects another header, adjust `call_api()`. The code expects a score field named `FINAL_ATS_MATCH_SCORE` (with a few fallback names); adjust if your API returns another name.

Output after API run: `dataset/score_comparison.csv`, including actual ATS score (parsed from percentage responses), expected rubric score and absolute error. The CSV also contains the AI feedback text when feedback is enabled; run with `--ai-feedback` to request it. Without that flag, the API intentionally skips AI feedback.
