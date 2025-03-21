# march-madness

A Python project for open-ended tournament management and simulation, specifically aimed at the NCAA March Madness tournament.

## Outline

TODO

## Usage

Detailed documentation available [here](https://github.com/harrisonfloam/march-madness/tree/main/src/march_madness).

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
    tournament.update_game_result(game=game, winner=game.team1)
```

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

## Setup

Clone the repository:
```bash
git clone https://github.com/harrisonfloam/march-madness.git
```

Activate the development environment:
```bash
conda env create -f environment.yml
conda activate march-madness
pip install -e .
```

Update the environment if any packages are added:

###### MacOS
```bash
conda env export --no-builds | grep -v '^prefix:' > environment.yml
```

###### Windows
```powershell
conda env export --no-builds | Select-String -NotMatch '^prefix:' | Out-File -Encoding utf8 environment.yml
```

## Dependencies

The only external dependency is [Ollama](https://ollama.com/download/windows), required for prediction with a locally installed LLM.