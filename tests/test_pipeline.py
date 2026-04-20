from ats_multiagent.models import JobDescription, Resume
from ats_multiagent.pipeline import MultiAgentATSPipeline


class StubLLM:
    def __init__(self):
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str, temperature: float = 0.2):
        self.calls += 1

        if "ATS planner agent" in user_prompt:
            return '["skill_match", "experience_depth"]'
        if "Evaluate ONE feature" in user_prompt:
            if "skill_match" in user_prompt:
                return '{"value": 0.9, "rationale": "Strong Python and SQL alignment."}'
            return '{"value": 0.7, "rationale": "5 years relevant experience."}'
        if "QA tester" in user_prompt:
            return "PASS: grounded in resume evidence"
        if "improvement agent" in user_prompt:
            return "Add quantified metrics extraction to next iteration."
        if "product critic and ideation agent" in user_prompt:
            return "- Blind spot: no culture-fit signal\n- Suggestion: add communication-quality feature"
        if "final ATS decision agent" in user_prompt:
            return '{"fit_score": 84, "strengths": ["Skill match", "Relevant tenure"], "concerns": ["No domain cert"], "final_recommendation": "SHORTLIST"}'
        raise AssertionError(f"Unexpected prompt: {user_prompt}")


def test_pipeline_runs_end_to_end():
    llm = StubLLM()
    pipeline = MultiAgentATSPipeline(llm)

    report = pipeline.screen(
        Resume(candidate_name="Jane Doe", text="Python SQL FastAPI 5 years"),
        JobDescription(title="ML Engineer", text="Need Python, SQL, APIs", required_skills=["Python", "SQL"]),
    )

    assert report.candidate_name == "Jane Doe"
    assert report.fit_score == 84
    assert report.final_recommendation == "SHORTLIST"
    assert len(report.feature_breakdown) == 2
    assert any(note.startswith("critic::") for note in report.iteration_notes)
