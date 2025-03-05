## Classes

### `Tournament`  
Manages a single-elimination tournament.

#### Parameters
- **teams** [pd.DataFrame | list[dict]]: Teams dataset, must contain at least the key (column) 'team'
- **initial_matchup_strategy** [Callable] = `ncaa_initial_matchups`: Generates first round matchups
- **round_matchup_strategy** [Callable] = `ncaa_round_matchups`: Generates second to final round matchups

#### Attributes
- **teams** [dict[str, dict]]: Dictionary mapping team names to team details
- **unplayed_games** [list[Game]]: List of games yet to be played  
- **played_games** [list[Game]]: List of completed games  
- **winner** [str, optional]: Name of the tournament winner  
- **num_teams**: Count of teams
- **num_rounds** [int]: Total number of rounds in the tournament, assuming a single-elimination tournament
- **num_games** [int]: Total number of games in the tournament  , assuming a single-elimination tournament

#### Methods

- **update_game_result(game, winner)**  
Records the winner of a game and updates tournament progress.  

    - **game** [Game]: The game to update
    - **winner** [str]: The winning team
    - **Raises** `ValueError` if the number of winners is incorrect  

- **get_unplayed_games(round_number=None)**  
Returns the list of unplayed games, optionally filtering by round.  

#### Example Usage

###### Create a Tournament instance
```python
import pandas as pd

from march_madness.tournament import Tournament
from march_madness.matchup import ncaa_initial_matchups, ncaa_round_matchups

teams = pd.read_csv("data/march_madness_2024.csv")

tournament = Tournament(teams=teams, 
                        initial_matchup_strategy=ncaa_initial_matchups, 
                        round_matchup_strategy=ncaa_round_matchups)
```

###### Extract the first round games
```python
for game in tournament.get_unplayed_games():
    print(f"Round {game.round_number}: {game.team1} vs {game.team2}")

    # Set team1 as the winner for each game
    tournament.update_game_result(game, game.team1)
```

---

### `TournamentSimulator`
Runs Monte Carlo simulations of a tournament.

#### Parameters
- **num_trials** [int]: Number of simulations to run
- **tournament_class** [type] = `Tournament`: Tournament class to simulate
- **tournament_params** [dict]: Params to initialize the tournament with
- **prediction_strategy** [Callable]: Generates gamer winner predictions
- **prediction_strategy_kwargs** [dict] = {}: Key word arguments to pass to the prediction strategy function
- **result_path** [str | Path] = None: Path to cache simulation results as csv
- **seed** [int] = 42: Ensures reproducibility across trials by seeding `np.random.Generator`. New seeds are generated for each prediction, creating stochasticity within trials.

#### Attributes
- **results** [list[dict]]: Stores simulation results
- **seeds** [list[int]]: List of seeds used to seed `np.random.Generator` for each trial

#### Methods
- **run**(resume=False, verbose=True)  
Runs multiple tournament simulations and logs results.

    - **resume** [bool]: If true, loads previous trials in `self.result_path` and ensures all new trial seeds are unique. Appends to existing csv.
    - **verbose** [bool]: Toggles `tqdm` progress bar visibility, with results logged at the game level

- **to_dataframe**()  
Returns the results as a Pandas DataFrame.

#### Example Usage

###### Simulate with a simple strategy
```python
from march_madness.prediction import team1_always_wins
from march_madness.tournament import Tournament, TournamentSimulator

simulator = TournamentSimulator(num_trials=100,
                                tournament_class=Tournament,
                                tournament_params={"teams": teams},
                                prediction_strategy=team1_always_wins,
                                seed=42)

simulator.run(verbose=True)
df = simulator.to_dataframe()   # Extract results
```

###### Simulate using an LLM to predict
```python
from march_madness.prediction import llm_prediction
# ...

simulator = TournamentSimulator(num_trials=100,
                                tournament_class=Tournament,
                                tournament_params={"teams": teams},
                                prediction_strategy=llm_prediction,
                                prediction_strategy_kwargs={
                                    "model_name": "llama3.2:1b"
                                },
                                result_path="results/llam3_2-1b.csv", # Add a cache directory
                                seed=42)
simulator.run(resume=True, verbose=True)  # Resume from previous trial
```