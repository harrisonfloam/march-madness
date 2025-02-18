from typing import List, Callable
from math import log2
import random
import logging

class Game:
    def __init__(self, team1: str, team2: str, round_number: int):
        self.team1 = team1
        self.team2 = team2
        self.round_number = round_number
        self.winner = None  # To be set when a result is available
    
    def set_winner(self, winner: str):
        self.winner = winner

class Tournament:
    def __init__(self, teams: List[str], region_sizes: List[int] = [16, 16, 16, 16], 
                 seeding_strategy: Callable = None):
        """
        Initializes a tournament.
        
        Args:
            teams: A list of team names.
            region_sizes: A list defining the size of each region. Default is 16 teams per region.
            seeding_strategy: A function that determines first-round matchups.
        """
        self.region_sizes = region_sizes #TODO: need to validate region_sizes
        self.rounds = {i: [] for i in range(1, self.num_rounds(len(teams)) + 1)}
        self.unplayed_games = []
        
        # Set default seeding strategy if none is provided
        if seeding_strategy is None:
            seeding_strategy = self.ncaa_seeding_strategy
        
        self.create_initial_matchups(teams, seeding_strategy)

    def num_rounds(self, num_teams: int) -> int:
        """Calculate the number of rounds in a single-elimination tournament."""
        return int(log2(num_teams))

    def create_initial_matchups(self, teams: List[str], seeding_strategy: Callable):
        """Creates first-round matchups using the provided seeding strategy."""
        #TODO: does this work with other region sizes...
        index = 0
        for region_size in self.region_sizes:
            region_teams = teams[index:index + region_size]
            index += region_size
            matchups = seeding_strategy(region_teams)
            
            for team1, team2 in matchups:
                game = Game(team1, team2, 1)
                self.rounds[1].append(game)
                self.unplayed_games.append(game)

    def update_game_result(self, game: Game, winner: str):
        """Updates a game's result and advances the winner to the next round."""
        game.set_winner(winner)
        self.unplayed_games.remove(game)

        next_round = game.round_number + 1
        if next_round <= self.num_rounds(len(self.rounds[1]) * 2):  # Prevent exceeding final round
            if len(self.rounds[next_round]) % 2 == 0:
                new_game = Game(winner, None, next_round)  # Placeholder for opponent
                self.rounds[next_round].append(new_game)
            else:
                self.rounds[next_round][-1].team2 = winner  # Assign opponent
                self.unplayed_games.append(self.rounds[next_round][-1])

    def get_unplayed_games(self) -> List[Game]:
        """Returns a list of games that have not yet been played."""
        return self.unplayed_games

    @staticmethod
    def ncaa_seeding_strategy(region_teams: List[str]) -> List[tuple]:
        """Generates first-round matchups using NCAA-style seeding."""
        size = len(region_teams)
        return [(region_teams[i], region_teams[size - 1 - i]) for i in range(size // 2)]

    @staticmethod
    def random_seeding_strategy(region_teams: List[str]) -> List[tuple]:
        """Generates first-round matchups using random seeding."""
        random.shuffle(region_teams)
        return [(region_teams[i], region_teams[i + 1]) for i in range(0, len(region_teams), 2)]

    #TODO:
    # print tournament state somehow
    # what is random seeding used for... shouldnt be static method