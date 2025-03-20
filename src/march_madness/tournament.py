"""Tournament class definition."""

from typing import List, Callable, Dict, Any, Tuple, Literal
from math import log2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import os

from march_madness.matchup import ncaa_initial_matchups, ncaa_round_matchups
from march_madness.types import Team, Matchup, Prediction


class Game:
    def __init__(self, team1: Team, team2: Team, round_number: int, game_id: Dict = None):
        for team in (team1, team2):
            if "team" not in team:
                raise ValueError(f"Team dictionary must contain at least key 'team'. Found keys: {team.keys()}")
        
        self.team1 = team1["team"]
        self.team2 = team2["team"]
        self.team1_details = team1
        self.team2_details = team2
        self.round_number = round_number
        self.game_id = game_id  # Not required
        self.winner = None
        self.winner_details = None
        
    def set_winner(self, winner: str):
        if winner not in {self.team1, self.team2}:
            raise ValueError(f"Winner must be either {self.team1} or {self.team2}, got '{winner}'.")
        self.winner = winner
        self.winner_details = self.team1_details if self.winner == self.team1 else self.team2_details

class Tournament:
    def __init__(self, teams: pd.DataFrame | List[Dict], 
                 initial_matchup_strategy: Callable[["Tournament"], List[Matchup]] = ncaa_initial_matchups,
                 round_matchup_strategy: Callable[["Tournament"], List[Matchup]] = ncaa_round_matchups):
        """
        Initializes a tournament.
        
        Args:
            teams: A structured input containing team data (can be a DataFrame or list of dicts).
            initial_matchup_strategy: A function that determines first-round matchups.
            round_matchup_strategy: A function that determines matchups in later rounds.
        """
        self.teams = self._process_teams(teams) #TODO: confusing that param teams isn't the same as attr teams
        self.initial_matchup_strategy = initial_matchup_strategy
        self.round_matchup_strategy = round_matchup_strategy
        
        self.unplayed_games = []
        self.played_games = []
        self.winner = None
        
        self.num_teams = len(teams)
        self.num_rounds = int(log2(self.num_teams)) # Single elimination tournament
        self.num_games = self.num_teams - 1
        
        self._validate_tournament()
        self._create_initial_matchups()
        
    def _validate_tournament(self):
        """Validates tournament parameters before proceeding."""
        num_teams = len(self.teams)
        if num_teams == 0 or (num_teams & (num_teams - 1)) != 0:    # Bitwise power of 2 check
            raise ValueError(f"Invalid number of teams: {num_teams}. Must be a power of 2 for a single-elimination tournament.")

        # Ensure all teams have required fields
        required_fields = {"team"}
        if self.initial_matchup_strategy == ncaa_initial_matchups:
            required_fields.update({"seed", "region"})  # NCAA-specific fields

        for team in self.teams.values():
            missing_fields = required_fields - team.keys()
            if missing_fields:
                raise ValueError(f"Missing required fields {missing_fields} in team entry: {team}")

                
    def _process_teams(self, teams: Any) -> Dict[str, Team]:
        """Converts a DataFrame or list of dicts into a standard format."""
        if isinstance(teams, pd.DataFrame):
            teams = teams.to_dict(orient="records")  # Convert DataFrame to list of dicts
        if isinstance(teams, list) and all(isinstance(t, dict) for t in teams):
            return {team["team"]: team for team in teams}
        else:
            raise ValueError("Teams must be provided as a DataFrame or a list of dictionaries.")

    def update_game_result(self, game: Game, winner: str):
        """Marks the winner of a game and builds the next round when all games are complete."""
        game.set_winner(winner)
        self.unplayed_games.remove(game)
        self.played_games.append(game)
        
        # If final round, set the tournament winner
        if self.current_round == self.num_rounds:
            self.winner = winner

        # If all games for this round are completed, advance the round
        elif len(self.unplayed_games) == 0:
            self._create_next_round()
    
    def get_unplayed_games(self, round_number: int = None) -> List[Game]:
        """Returns unplayed games, optionally filtering by round."""
        if round_number:
            return [game for game in self.unplayed_games if game.round_number == round_number]
        return self.unplayed_games.copy()   # Return as copy so future changes don't affect iterable use

    def _create_initial_matchups(self):
        """Creates first-round matchups using the initial matchup strategy."""
        self.current_round = 1
        matchups = self.initial_matchup_strategy(self)
        if not matchups:
            raise ValueError("Initial matchup strategy did not return any games. Check the strategy function.")
        
        for team1, team2, game_id in matchups:
            game = Game(team1, team2, 1, game_id)
            self.unplayed_games.append(game)

    def _create_next_round(self):
        """Creates matchups for the next round using the round matchup strategy."""
        winners = [game.winner for game in self.played_games if game.winner and game.round_number == self.current_round]

        # If odd number of winners, tournament is broken.
        if len(winners) % 2 != 0:
            raise ValueError(f"Found {len(winners)} winners from round {self.current_round}; Expected even count.")

        self.current_round += 1 # Advance round counter
        matchups = self.round_matchup_strategy(self)

        # Create new games for the next round
        new_games = [Game(team1, team2, self.current_round, game_id) for team1, team2, game_id in matchups]
        self.unplayed_games.extend(new_games)
        
    def get_tournament_state(self):
        return [
            {
                "game_id": game.game_id,
                "winner": game.winner
            }
            for game in self.played_games
        ]

class TournamentSimulator:
    def __init__(self, num_trials: int, 
                 prediction_strategy: Callable[["Game", "Tournament", np.random.Generator, Any], Prediction], 
                 tournament_params: Dict[str, Any], tournament_class: type = Tournament, 
                 prediction_strategy_params: Dict[str, Any] = {},
                 result_path: str | Path = None,
                 seed: int = 42):
        #TODO: make tournament_class simpler... instantiate before?
        """
        Monte Carlo tournament simulator.

        Args:
            num_trials: Number of tournament simulations to run.
            tournament_class: The Tournament class to instantiate.
            tournament_params: Dictionary of parameters to initialize Tournament.
            prediction_strategy: Callable that predicts the winner of a game.
            prediction_strategy_params: Dictionary of parameters to pass to prediction_strategy callable.
            result_path: Path to .csv to cache results. Loaded when calling `run` with `resume=True`.
        """
        self.num_trials = num_trials
        self.tournament_class = tournament_class
        self.tournament_params = tournament_params
        self.prediction_strategy = prediction_strategy
        self.prediction_strategy_params = prediction_strategy_params
        self.result_path = Path(result_path) if result_path else None
        
        # RNG
        self.seed = seed
        
        self.results = []

    def run(self, resume: bool = False, verbose: bool = True, pbar_level: Literal["game", "trial"] = "game"):
        """Runs multiple tournament simulations and logs results."""
        if pbar_level not in ["game", "trial"]:
            raise ValueError(f"pbar_level must be either 'game' or 'trial', got {pbar_level}.")
        
        existing_seeds = None
        if self.result_path:
            os.makedirs(os.path.dirname(self.result_path), exist_ok=True)
            if resume:
                # Don't use the existing seeds again
                existing_seeds = self._load_existing_results()
            else:
                # Overwrite the existing results
                if os.path.exists(self.result_path):
                    os.remove(self.result_path)
                
        trial_seeds = self._generate_seeds(exclude=existing_seeds)
        trials = zip(range(1, self.num_trials+1), trial_seeds)
        tournament = self.tournament_class(**self.tournament_params)
        total_steps = self.num_trials * tournament.num_games
        
        pbar = tqdm(trials, 
                    total=total_steps, 
                    postfix={
                        "trial": 1,
                        "game_winner": None,
                        "tournament_winner": None
                    },
                    disable=not verbose)
        prev_tournament_winner = None
        for trial, trial_seed in trials:
            tournament = self.tournament_class(**self.tournament_params)
            rng = np.random.default_rng(trial_seed)  # One RNG per trial
            trial_results = []
            
            while tournament.get_unplayed_games():
                for game in tournament.get_unplayed_games():
                    predicted_winner, prediction_details = self.prediction_strategy(tournament, game, rng, **self.prediction_strategy_params)
                    tournament.update_game_result(game, predicted_winner)
                    
                    # Clean up result keys
                    prefixed_prediction_details = {f"prediction_{k}": v for k, v in (prediction_details or {}).items()}
                    prefixed_team1_details = {f"team1_{k}": v for k, v in game.team1_details.items() if k != "team"}
                    prefixed_team2_details = {f"team2_{k}": v for k, v in game.team2_details.items() if k != "team"}
                    prefixed_winner_details = {f"winner_{k}": v for k, v in game.winner_details.items() if k != "team"}
                    
                    result = {
                        "trial": trial,
                        "trial_seed": trial_seed,
                        "game_id": game.game_id,
                        "round": game.round_number,
                        "team1": game.team1,
                        "team2": game.team2,
                        "game_winner": predicted_winner,
                        "tournament_winner": tournament.winner,
                        **prefixed_prediction_details,
                        **prefixed_team1_details,
                        **prefixed_team2_details,
                        **prefixed_winner_details,
                        "tournament_state": tournament.get_tournament_state(),
                    }
                    trial_results.append(result)

                    pbar.update()
                    if pbar_level == "game":
                        pbar.set_postfix({
                            "trial": trial,
                            "team1": game.team1,
                            "team2": game.team2,
                            "game_winner": predicted_winner,
                            "tournament_winner": tournament.winner,
                            "prev_tournament_winner": prev_tournament_winner
                        })
                    if tournament.winner:
                        if pbar_level == "trial":
                            pbar.set_postfix({
                                "trial": trial,
                                "tournament_winner": tournament.winner
                            })
                        prev_tournament_winner = tournament.winner
                
            self._update_results(trial_results)

    def _generate_seeds(self, exclude: set = None) -> List[int]:
        """Generate random seeds for each trial for reproducibility."""
        seed_max = 2**32 - 1
        exclude = exclude or set()
        seeds = set()  # Stores new unique seeds
        while len(seeds) < self.num_trials:
            new_seed = np.random.randint(0, seed_max)
            if new_seed not in exclude and new_seed not in seeds:
                seeds.add(new_seed)

        return list(seeds)
        
    def _update_results(self, trial_results):
        self.results.extend(trial_results)
        # Cache results
        if self.result_path:
            df = pd.DataFrame(trial_results)
            df.to_csv(self.result_path, mode="a", header=not self.result_path.exists(), index=False)
        
    def _load_existing_results(self) -> set:
        """Load trial seeds from existing results to avoid duplication."""
        if self.result_path.exists():
            df = pd.read_csv(self.result_path, usecols=["trial_seed"])
            return set(df["trial_seed"].dropna().astype(int))
        else:
            return set()
    
    def to_dataframe(self):
        """Returns the logged results as a Pandas DataFrame."""
        return pd.DataFrame(self.results)
    