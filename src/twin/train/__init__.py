"""Self-play training loop (DESIGN.md §5)."""

from twin.train.extract import count_oracle_calls, extract_final_answer
from twin.train.loop import SelfPlayTrainer

__all__ = ["SelfPlayTrainer", "extract_final_answer", "count_oracle_calls"]
