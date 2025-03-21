## Table of Contents
- [Table of Contents](#table-of-contents)
- [Classes](#classes)
  - [`Tournament`](#tournament)
    - [Parameters](#parameters)
    - [Attributes](#attributes)
    - [Methods](#methods)
    - [Example Usage](#example-usage)
        - [Create a Tournament instance](#create-a-tournament-instance)
        - [Extract the first round games](#extract-the-first-round-games)
  - [`TournamentSimulator`](#tournamentsimulator)
    - [Parameters](#parameters-1)
    - [Attributes](#attributes-1)
    - [Methods](#methods-1)
    - [Example Usage](#example-usage-1)
        - [Simulate with a simple strategy](#simulate-with-a-simple-strategy)
        - [Simulate using an LLM to predict](#simulate-using-an-llm-to-predict)
  - [`Game`](#game)
    - [Parameters](#parameters-2)
    - [Attributes](#attributes-2)
    - [Methods](#methods-2)
- [Prediction Strategies](#prediction-strategies)
  - [`random_prediction`](#random_prediction)
    - [Parameters](#parameters-3)
    - [Returns](#returns)
  - [`team1_always_wins`](#team1_always_wins)
    - [Parameters](#parameters-4)
    - [Returns](#returns-1)
  - [`llm_prediction`](#llm_prediction)
    - [Parameters](#parameters-5)
    - [Returns](#returns-2)
- [Matchup Strategies](#matchup-strategies)
  - [`ncaa_initial_matchups`](#ncaa_initial_matchups)
    - [Parameters](#parameters-6)
    - [Returns](#returns-3)
    - [Description](#description)
  - [`ncaa_round_matchups`](#ncaa_round_matchups)
    - [Parameters](#parameters-7)
    - [Returns](#returns-4)
    - [Description](#description-1)

## Classes 

### `Tournament`   
*march_madness.tournament.Tournament*

Manages a single-elimination tournament.

#### Parameters
- **team_data** [pd.DataFrame | PathLike | list[dict]]: Teams dataset, must contain at least the key (column) 'team'; may pass filepath to load from csv internally
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
from march_madness.tournament import Tournament
from march_madness.matchup import ncaa_initial_matchups, ncaa_round_matchups

tournament = Tournament(team_data="data/march_madness_2024.csv", 
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
*march_madness.tournament.TournamentSimulator*

Runs Monte Carlo simulations of a tournament.

#### Parameters
- **num_trials** [int]: Number of simulations to run
- **tournament_class** [type] = `Tournament`: Tournament class to simulate
- **tournament_params** [dict]: Tournament initialization params
- **prediction_strategy** [Callable]: Generates game winner predictions
- **prediction_strategy_params** [dict] = {}: Params to pass to the prediction strategy function
- **result_path** [str | Path] = None: Path to cache simulation results as csv
- **seed** [int] = 42: Ensures reproducibility across trials by seeding `np.random.Generator`. New seeds are generated for each prediction, creating stochasticity within trials.

#### Attributes
- **results** [list[dict]]: Stores simulation results
- **seeds** [list[int]]: List of seeds used to seed `np.random.Generator` for each trial

#### Methods
- **run**(resume=False, verbose=True)  
Runs multiple tournament simulations and logs results.

    - **resume** [bool]: If true, loads previous trials in `self.result_path` and ensures all new trial seeds are unique. Appends to existing csv.
    - **verbose** [bool]: Toggles `tqdm` progress bar visibility
    - **pbar_level** [str]: "game" or "trial"; controls the level of detail included in the pbar postfix

- **to_dataframe**()  
Returns the results as a Pandas DataFrame.

#### Example Usage

###### Simulate with a simple strategy
```python
from march_madness.prediction import team1_always_wins
from march_madness.tournament import Tournament, TournamentSimulator

simulator = TournamentSimulator(num_trials=100,
                                tournament_class=Tournament,
                                tournament_params={
                                    "team_data": "data/march_madness_2024.csv"
                                },
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
                                tournament_params={
                                    "team_data": "data/march_madness_2024.csv"
                                },
                                prediction_strategy=llm_prediction,
                                prediction_strategy_params={
                                    "model_name": "llama3.2:1b"
                                },
                                result_path="results/llam3_2-1b.csv", # Add a cache directory
                                seed=42)
simulator.run(resume=True, verbose=True)  # Resume from previous trial
```

---

### `Game`
*march_madness.tournament.Game*

Represents a single matchup within the tournament bracket.

#### Parameters
- **team1** [dict]: A dictionary representing the first team; must include at least the key `'team'`.
- **team2** [dict]: A dictionary representing the second team; must include at least the key `'team'`.
- **round_number** [int]: The round in which this game is played.
- **game_id** [int, optional]: A unique identifier for the game (if provided).

#### Attributes
- **team1** [str]: Name of the first team.
- **team2** [str]: Name of the second team.
- **team1_details** [dict]: Complete details for the first team.
- **team2_details** [dict]: Complete details for the second team.
- **round_number** [int]: The tournament round number for the game.
- **game_id** [int, optional]: Unique game identifier.
- **winner** [str, optional]: The name of the winning team (set after the game is played).
- **winner_details** [dict, optional]: Details of the winning team (populated once a winner is set).

#### Methods
- **set_winner(winner: str)**  
Sets the winner for the game, ensuring that the specified team is one of the competing teams.
    - **winner** [str]: The team name to be set as the winner.
    - **Raises** `ValueError` if the provided `winner` does not match either team.

## Prediction Strategies

### `random_prediction`
*march_madness.prediction.random_prediction*

Randomly selects a game winner based on a specified probability.
  
#### Parameters
- **tournament** [Tournament]: The tournament instance.
- **game** [Game]: The game for which the prediction is made.
- **rng** [np.random.Generator]: Random number generator for stochastic outcomes.
- **p_team1** [float] (default=0.50): The probability of selecting team1 as the winner.
  
#### Returns
- **Prediction**: A tuple containing the predicted winning team (str) and a dictionary of prediction details (e.g. confidence and reasoning).

---

### `team1_always_wins`
*march_madness.prediction.team1_always_wins*

Always predicts that team1 wins the game.
  
#### Parameters
- **tournament** [Tournament]: The tournament instance.
- **game** [Game]: The game for which the prediction is made.
- **rng** [np.random.Generator]: Random number generator (unused in this strategy).
  
#### Returns
- **Prediction**: A tuple containing team1 as the winning team (str) and fixed prediction details.

---

### `llm_prediction`
*march_madness.prediction.llm_prediction*

Uses an LLM to generate a game prediction. Constructs a prompt based on team details and tournament state.
  
#### Parameters
- **tournament** [Tournament]: The tournament instance.
- **game** [Game]: The game for which the prediction is made.
- **rng** [np.random.Generator]: Random number generator for generating a unique seed per prediction.
- **model_name** [str]: Specifies the LLM model to use (e.g., OpenAI's GPT variants or Ollama models).
  
#### Returns
- **Prediction**: A tuple containing the predicted winning team (str) and a dictionary of prediction details.

## Matchup Strategies

### `ncaa_initial_matchups`
*march_madness.matchup.ncaa_initial_matchups*

Generates first-round matchups using NCAA-style seeding.

#### Parameters
- **tournament** [Tournament]: The tournament instance containing team data and configuration.

#### Returns
- **List[Matchup]**: A list of matchup tuples, where each tuple consists of two team dictionaries paired for the game and a unique game identifier.

#### Description
In each region, teams are paired such that the highest seed plays the lowest, the second-highest plays the second-lowest, and so on, following the NCAA bracket seeding defined by the `NCAA_BRACKET_PROGRESSION` order.

---

### `ncaa_round_matchups`
*march_madness.matchup.ncaa_round_matchups*

Generates next-round matchups using NCAA-style progression.

#### Parameters
- **tournament** [Tournament]: The tournament instance containing the results of the previous round and team data.

#### Returns
- **List[Matchup]**: A list of matchup tuples, where each tuple consists of the two teams paired for the next round and a unique game identifier.

#### Description
Based on the winners of the previous round:
- **Regional Rounds:** Winners within each region are paired according to the NCAA bracket structure.
- **Final Four:** Winners are paired across regions (e.g., East vs. West and South vs. Midwest).
- **Championship:** The remaining winners are paired for the final matchup.
