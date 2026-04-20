# Multi-Agent ATS Resume Screening (Groq API)

This project implements a **multi-agent ATS (Applicant Tracking System)** for resume screening with a strict:

- **Plan → Execute → Test → Improve** loop
- **Critique + idea generation** after each new feature addition

## Architecture

Agents:
1. **PlannerAgent**: creates ordered feature plan.
2. **FeatureExecutionAgent**: evaluates each feature against resume/job.
3. **TestAgent**: validates feature quality and grounding.
4. **ImprovementAgent**: proposes correction/improvement action.
5. **CriticAgent**: critiques blind spots and suggests next feature idea after each addition.
6. **SynthesisAgent**: produces final fit score and recommendation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
export GROQ_API_KEY="your_key"
```

## Usage

```bash
ats-run \
  --resume ./examples/resume.txt \
  --job ./examples/job.txt \
  --name "Jane Doe" \
  --skills "python,sql,llm,fastapi"
```

The CLI prints a JSON screening report with feature breakdown + iteration notes.

## Notes

- Uses Groq-compatible OpenAI chat endpoint: `/chat/completions`.
- Keep prompts returning JSON where parsing is required.
