from __future__ import annotations

import json
from dataclasses import dataclass

from .models import FeatureResult, JobDescription, Resume


@dataclass
class AgentContext:
    resume: Resume
    job: JobDescription
    prior_notes: list[str]


class BaseAgent:
    role: str = "generic"

    def __init__(self, llm_client):
        self.llm = llm_client

    def run(self, context: AgentContext):
        raise NotImplementedError


class PlannerAgent(BaseAgent):
    role = "planner"

    def run(self, context: AgentContext) -> list[str]:
        prompt = f"""
You are an ATS planner agent.
Create a short, ordered feature plan to evaluate resume-to-job fit.
Return strict JSON array of strings.

Job Description:
{context.job.text}

Required Skills:
{", ".join(context.job.required_skills)}
"""
        raw = self.llm.complete(system_prompt="Respond only in JSON.", user_prompt=prompt)
        plan = json.loads(raw)
        if not isinstance(plan, list):
            raise ValueError("Planner output must be a list")
        return [str(item) for item in plan]


class FeatureExecutionAgent(BaseAgent):
    role = "executor"

    def run(self, context: AgentContext, feature_name: str) -> FeatureResult:
        prompt = f"""
You are an ATS execution agent.
Evaluate ONE feature for candidate screening.
Return strict JSON object with keys: value, rationale.

Feature: {feature_name}
Resume:
{context.resume.text}

Job Description:
{context.job.text}

Existing notes:
{context.prior_notes}
"""
        raw = self.llm.complete(system_prompt="Respond only in JSON.", user_prompt=prompt)
        data = json.loads(raw)
        return FeatureResult(
            name=feature_name,
            value=data.get("value"),
            rationale=str(data.get("rationale", "")),
        )


class TestAgent(BaseAgent):
    role = "tester"

    def run(self, context: AgentContext, feature_result: FeatureResult) -> str:
        prompt = f"""
You are a QA tester for ATS logic.
Check if the feature analysis is grounded in resume evidence and not generic.
Return one concise line: PASS or FAIL, followed by reason.

Feature result:
{feature_result}

Resume:
{context.resume.text}
"""
        return self.llm.complete(system_prompt="Be strict and concise.", user_prompt=prompt, temperature=0)


class ImprovementAgent(BaseAgent):
    role = "improver"

    def run(self, context: AgentContext, feature_result: FeatureResult, test_feedback: str) -> str:
        prompt = f"""
You are an improvement agent.
Given feature output and QA feedback, produce one improvement action for future iterations.

Feature result:
{feature_result}
QA feedback:
{test_feedback}
"""
        return self.llm.complete(system_prompt="Return one concise action line.", user_prompt=prompt)


class CriticAgent(BaseAgent):
    role = "critic"

    def run(self, context: AgentContext, newly_added_feature: FeatureResult) -> str:
        prompt = f"""
You are a product critic and ideation agent for ATS systems.
After every new feature addition, provide:
1) one critique of blind spots
2) one suggestion for next feature addition.

Newly added feature:
{newly_added_feature}
Current notes:
{context.prior_notes}
"""
        return self.llm.complete(system_prompt="Return 2 bullet points only.", user_prompt=prompt)


class SynthesisAgent(BaseAgent):
    role = "synthesizer"

    def run(self, context: AgentContext, features: list[FeatureResult], notes: list[str]):
        prompt = f"""
You are a final ATS decision agent.
Create final decision JSON with keys:
fit_score (0-100), strengths (array), concerns (array), final_recommendation (string)

Features:
{features}

Iteration notes:
{notes}
"""
        raw = self.llm.complete(system_prompt="Respond only in JSON.", user_prompt=prompt)
        return json.loads(raw)
