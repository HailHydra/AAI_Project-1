from __future__ import annotations

from .agents import (
    AgentContext,
    CriticAgent,
    FeatureExecutionAgent,
    ImprovementAgent,
    PlannerAgent,
    SynthesisAgent,
    TestAgent,
)
from .models import JobDescription, Resume, ScreeningReport


class MultiAgentATSPipeline:
    def __init__(self, llm_client):
        self.planner = PlannerAgent(llm_client)
        self.executor = FeatureExecutionAgent(llm_client)
        self.tester = TestAgent(llm_client)
        self.improver = ImprovementAgent(llm_client)
        self.critic = CriticAgent(llm_client)
        self.synthesizer = SynthesisAgent(llm_client)

    def screen(self, resume: Resume, job: JobDescription, max_features: int = 6) -> ScreeningReport:
        notes: list[str] = []
        context = AgentContext(resume=resume, job=job, prior_notes=notes)

        plan = self.planner.run(context)[:max_features]
        features = []

        for feature_name in plan:
            feature_result = self.executor.run(context, feature_name)
            features.append(feature_result)

            qa_feedback = self.tester.run(context, feature_result)
            notes.append(f"test::{feature_name}::{qa_feedback}")

            improvement_action = self.improver.run(context, feature_result, qa_feedback)
            notes.append(f"improve::{feature_name}::{improvement_action}")

            critique_and_next_idea = self.critic.run(context, feature_result)
            notes.append(f"critic::{feature_name}::{critique_and_next_idea}")

        final = self.synthesizer.run(context, features, notes)
        return ScreeningReport(
            candidate_name=resume.candidate_name,
            fit_score=float(final.get("fit_score", 0)),
            strengths=[str(s) for s in final.get("strengths", [])],
            concerns=[str(c) for c in final.get("concerns", [])],
            final_recommendation=str(final.get("final_recommendation", "HOLD")),
            feature_breakdown=features,
            iteration_notes=notes,
        )
