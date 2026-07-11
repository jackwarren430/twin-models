"""21-questions mode (twentyq/DESIGN.md): creator sets secrets + answers,
solver asks questions across multiple turns, base judge validates/audits/scores."""

from twin.games.twentyq.schema import Secret, SecretParseError, guess_matches, parse_secret

__all__ = ["Secret", "SecretParseError", "guess_matches", "parse_secret"]
