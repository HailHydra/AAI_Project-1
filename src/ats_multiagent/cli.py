from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import Settings
from .groq_client import GroqClient
from .models import JobDescription, Resume
from .pipeline import MultiAgentATSPipeline


def load_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-agent ATS screening using Groq API")
    parser.add_argument("--resume", required=True, help="Path to candidate resume text file")
    parser.add_argument("--job", required=True, help="Path to job description text file")
    parser.add_argument("--name", required=True, help="Candidate name")
    parser.add_argument("--skills", default="", help="Comma-separated required skills")
    args = parser.parse_args()

    settings = Settings.from_env()
    client = GroqClient(settings)
    pipeline = MultiAgentATSPipeline(client)

    resume = Resume(candidate_name=args.name, text=load_text(args.resume))
    job = JobDescription(
        title="Target Role",
        text=load_text(args.job),
        required_skills=[s.strip() for s in args.skills.split(",") if s.strip()],
    )
    report = pipeline.screen(resume, job)

    print(json.dumps(report.__dict__, default=lambda obj: obj.__dict__, indent=2))


if __name__ == "__main__":
    main()
