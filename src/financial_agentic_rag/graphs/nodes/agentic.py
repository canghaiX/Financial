"""Compatibility layer for LangGraph node callables.

The concrete agent implementations live under financial_agentic_rag.agents.
"""

from financial_agentic_rag.agents import (
    executer,
    planner,
    router,
    simple_answer,
    synthesizer,
    verifier,
)
from financial_agentic_rag.agents.common import complete_answer as _complete_answer


__all__ = [
    "executer",
    "planner",
    "router",
    "simple_answer",
    "synthesizer",
    "verifier",
    "_complete_answer",
]
