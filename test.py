import requests

JD = """Senior Software Engineer, Backend (Cost Engineering)
Affirm - Remote Canada

4+ years of experience designing, developing and launching backend systems at scale using Python or Kotlin.
Track record of developing highly available distributed systems using AWS, MySQL and Kubernetes.
Experience supporting scale up/down of large scale sales events such as Prime Day, Black Friday.
Tuned autoscaling, CPU/Memory, and multi-threading for compute type workloads.
Strong verbal and written communication skills."""

response = requests.post(
    "http://localhost:8001/analyze",
    headers={"X-API-Key": "ats-2026"},
    json={
        "drive_links": [
            "https://drive.google.com/file/d/1aQIW2crNqEc7UrKFrd_G8PC3BEn55RD-/view?usp=sharing"
        ],
        "job_description": JD,
    },
    timeout=900,
)

print("Status:", response.status_code)
import json
print(json.dumps(response.json(), indent=2))
