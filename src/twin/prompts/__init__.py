"""Prompt templates (DESIGN.md §5, §9)."""

from twin.prompts.templates import (
    CREATOR_SYSTEM,
    JUDGE_SYSTEM,
    SOLVER_SYSTEM,
    THEMES,
    creator_user,
    pick_theme,
    solver_user,
)

__all__ = [
    "CREATOR_SYSTEM",
    "JUDGE_SYSTEM",
    "SOLVER_SYSTEM",
    "THEMES",
    "creator_user",
    "solver_user",
    "pick_theme",
]
