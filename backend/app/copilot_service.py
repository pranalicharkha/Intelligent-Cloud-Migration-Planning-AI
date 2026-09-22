from __future__ import annotations

from .models import CopilotResponse


def answer_copilot(question: str) -> CopilotResponse:
    normalized = (question or "").strip()
    lower = normalized.lower()

    if "wave" in lower or "sequence" in lower:
        answer = (
            "Sequence migration waves by starting with low-risk, dependency-light applications, "
            "then move to dependency-heavy and business-critical workloads once the foundation is ready."
        )
    elif "risk" in lower:
        answer = "Prioritize lower-risk applications first, then validate performance and dependency readiness before moving critical workloads."
    else:
        answer = (
            "Use the application portfolio to evaluate dependency exposure, criticality, and modernization effort before ordering workloads into migration waves."
        )

    return CopilotResponse(
        question=normalized,
        answer=answer,
        sources=["application_portfolio_1000.csv", "docs/ARCHITECTURE.md", "local migration playbook"],
    )
