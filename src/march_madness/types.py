"""Type definitions."""

from typing import List, Callable, Dict, Any, Tuple


Team = Dict[str, Any]  # Team dictionary
Matchup = Tuple[Team, Team]  # Matchup of teams
Prediction = Tuple[str, Dict[str, Any] | None]
