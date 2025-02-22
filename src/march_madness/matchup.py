"""Team matchup strategies."""

from typing import List, TYPE_CHECKING
import numpy as np
import pandas as pd
import itertools

if TYPE_CHECKING:
    from march_madness.tournament import Tournament, Game
from march_madness.types import Matchup


def ncaa_initial_matchups(tournament: "Tournament") -> List[Matchup]:
    """
    Generates first-round matchups using NCAA-style seeding.
    
    In each region, the highest seed plays the lowest, second-highest plays second-lowest: 
    1 vs. 16, 2 vs. 15, etc.
    """
    matchups = []
    regions = set(team["region"] for team in tournament.teams.values())  # All original regions

    for region in regions:
        region_teams = sorted(
            [team for team in tournament.teams.values() if team["region"] == region],
            key=lambda x: x["seed"]
        )
        for i in range(len(region_teams) // 2):
            matchups.append((region_teams[i], region_teams[-(i + 1)]))  # Highest vs. Lowest
    return matchups


def ncaa_round_matchups(tournament: "Tournament") -> List[Matchup]:
    """
    Generates next-round matchups using NCAA-style progression.
    
    Winners are paired in order defined by the bracket structure: 
    - Winner of (1 vs. 16) faces winner of (8 vs. 9)
    - Winner of (5 vs. 12) faces winner of (4 vs. 13), etc.
    - East plays West, South plays Midwest
    """
    # NCAA-defined bracket structure for regional rounds
    ncaa_bracket_progression = [
        1, 16, 8, 9,
        5, 12, 4, 13,
        6, 11, 3, 14,
        7, 10, 2, 15
    ]
    # Final Four region matchups
    inter_region_matchups = [
        ("East", "West"),
        ("South", "Midwest")
    ]

    # Get winners from last round
    prev_round_games = [game for game in tournament.played_games if game.winner and game.round_number == tournament.current_round-1]
    num_winners = len(prev_round_games)
    regions = set(team["region"] for team in tournament.teams.values())  # All original regions
    matchups = []

    # Regional rounds
    if num_winners > len(regions):
        for region in regions:
            region_winners = [tournament.teams[game.winner] for game in prev_round_games if game.winner_details["region"] == region]
            
            # Iterate over chunks of the bracket
            # In round 1, winners must come from seeds {16, 1, 8, 9}...
            for round_bracket in itertools.batched(ncaa_bracket_progression, 2**tournament.current_round):
                round_bracket_set = set(round_bracket)
                matchup = tuple(team for team in region_winners if team["seed"] in round_bracket_set)
                matchups.append(tuple(matchup))
                
    # Final Four
    elif num_winners == len(regions):
        winners_by_region = {tournament.teams[game.winner]["region"]: tournament.teams[game.winner] for game in prev_round_games}
        for region1, region2 in inter_region_matchups:
            matchups.append((winners_by_region[region1], winners_by_region[region2]))
            
    # Championship
    else:
        winners = tuple(tournament.teams[game.winner] for game in prev_round_games)
        matchups.append(winners)

    return matchups