"""Scoreboard package — score aggregation and formatting."""
from eval.scoreboard.aggregator import aggregate_scores
from eval.scoreboard.formatter import to_csv, to_markdown, to_pretty_text
from eval.scoreboard.types import ScoreRecord

__all__ = ["aggregate_scores", "to_csv", "to_markdown", "to_pretty_text", "ScoreRecord"]
