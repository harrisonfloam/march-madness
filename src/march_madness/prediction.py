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

def llm_prediction(tournament: "Tournament", game: "Game", rng: np.random.Generator) -> Prediction:
    seed = rng.integers(1, 2**32)
    
    prompt = """
        You are a NCAA college basketball expert. You can make predictions regarding the outcome of future games and accurately state your confidence level and reasoning. The 2025 NCAA March Madness tournament is beginning soon nd you are tasked with predicting its outcomes.

        When evaluating a certain game, thoroughly consider these key factors (and any other facts you deem relevant):
        - Historical Matchups & Tournament Trends: Review past meetings, tournament performances, and trends for similar teams.  
        - Team Details as of DATE: Evaluate rosters, injuries, coaching strategies, and key player changes from previous seasons.  
        - Tournament Performance & Momentum: Assess how teams have played in earlier rounds, including upsets, fatigue, and adaptability.  
        - Regular Season Performance & Strength of Schedule: Consider key wins, conference difficulty, and performance against top opponents.  
        - Tactical Matchups & Game Dynamics: Analyze team strengths, weaknesses, player matchups, and coaching strategies under pressure.  

        Respond **only** in this JSON format:
        {
            "winner": "Duke",
            "confidence": 0.50,
            "reasoning": "Duke always wins this matchup..."
        }

        Current matchup:
            Teams: Duke, UConn
            Round: 3
            
        Current tournament state:

        Historical context:

        """
    return