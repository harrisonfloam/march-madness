"""Team matchup strategies."""

from typing import List, TYPE_CHECKING
import numpy as np
import pandas as pd
import itertools

if TYPE_CHECKING:
    from march_madness.tournament import Tournament, Game
from march_madness.types import Matchup


# NCAA-defined bracket structure for regional rounds
NCAA_BRACKET_PROGRESSION = [
    1, 16, 8, 9,
    5, 12, 4, 13,
    6, 11, 3, 14,
    7, 10, 2, 15
]

NCAA_REGION_ORDER = ["East", "West", "South", "Midwest"]

def ncaa_initial_matchups(tournament: "Tournament") -> List[Matchup]:
    """
    Generates first-round matchups using NCAA-style seeding.
    
    In each region, the highest seed plays the lowest, second-highest plays second-lowest: 
    1 vs. 16, 2 vs. 15, etc.
    """
    game_id = 0
    matchups = []
    for region in NCAA_REGION_ORDER:
        teams_by_seed = {team["seed"]: team for team in tournament.teams.values() if team["region"] == region}

        for (seed1, seed2) in itertools.batched(NCAA_BRACKET_PROGRESSION, 2):
            game_id += 1

            matchups.append((teams_by_seed[seed1], teams_by_seed[seed2], game_id))  # Highest vs. Lowest
    return matchups

def ncaa_round_matchups(tournament: "Tournament") -> List[Matchup]:
    """
    Generates next-round matchups using NCAA-style progression.
    
    Winners are paired in order defined by the bracket structure: 
    - Winner of (1 vs. 16) faces winner of (8 vs. 9)
    - Winner of (5 vs. 12) faces winner of (4 vs. 13), etc.
    - East plays West, South plays Midwest
    """

    # Get winners from last round
    prev_round_games = [game for game in tournament.played_games if game.winner and game.round_number == tournament.current_round-1]
    num_winners = len(prev_round_games)
    num_regions = len(NCAA_REGION_ORDER)
    game_id = int(sum(64/2**r for r in range(1, tournament.current_round)))
    matchups = []

    # Regional rounds
    if num_winners > num_regions:
        for region in NCAA_REGION_ORDER:
            region_winners = [tournament.teams[game.winner] for game in prev_round_games if game.winner_details["region"] == region]
            
            # Iterate over chunks of the bracket
            # In round 1, winners must come from seeds {16, 1, 8, 9}...
            for round_bracket in itertools.batched(NCAA_BRACKET_PROGRESSION, 2**tournament.current_round):
                round_bracket_set = set(round_bracket)
                matchup = tuple(team for team in region_winners if team["seed"] in round_bracket_set)
                game_id += 1
                matchups.append((*matchup, game_id))
                
    # Final Four
    elif num_winners == num_regions:
        winners_by_region = {tournament.teams[game.winner]["region"]: tournament.teams[game.winner] for game in prev_round_games}
        for region1, region2 in zip(NCAA_REGION_ORDER[:2], NCAA_REGION_ORDER[2:]):
            game_id += 1
            matchups.append((winners_by_region[region1], winners_by_region[region2], game_id))
            
    # Championship
    else:
        winners = tuple(tournament.teams[game.winner] for game in prev_round_games)
        game_id += 1
        matchups.append((*winners, game_id))

    return matchups