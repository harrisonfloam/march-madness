"""Tournament prediction classes and methods."""

from typing import TYPE_CHECKING
import numpy as np


if TYPE_CHECKING:
    from march_madness.tournament import Tournament, Game
from march_madness.types import Prediction


def random_prediction(tournament: "Tournament", game: "Game", rng: np.random.Generator) -> Prediction:
    """Randomly picks one of the teams as the winner."""
    winner = str(rng.choice([game.team1, game.team2]))
    return winner, {"confidence": 0.50}

def team1_always_wins(tournament: "Tournament", game: "Game", rng: np.random.Generator) -> Prediction:
    """Always selects team1 as the winner."""
    return game.team1, None
