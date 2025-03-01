# march-madness

A Python project for open-ended tournament management and simulation, specifically aimed at the NCAA March Madness tournament.

## Outline

- `src/march_madness/`: main project directory
  - tournament.py: 
  - prediction.py: 
  - matchup.py: 
  - types.py: 
- `notebooks/`: tutorial notebooks
  - march_madness.ipynb: 
- `data/`: .csv files containing tournament initialization data
  - march_madness_2024.csv

## Usage

Detailed documentation available [here](https://github.com/harrisonfloam/march-madness/blob/main/src/march_madness/README.md).

###### Create a Tournament instance
```python
import pandas as pd

from march_madness.tournament import Tournament
from march_madness.matchup import ncaa_initial_matchups, ncaa_round_matchups

teams = pd.read_csv("data/march_madness_2024.csv")

tournament = Tournament(teams, ncaa_initial_matchups, ncaa_round_matchups)
```

###### Extract the first round games
```python
for game in tournament.get_unplayed_games():
    print(f"Round {game.round_number}: {game.team1} vs {game.team2}")

    # Set team1 as the winner for each game
    tournament.update_game_result(game, game.team1)
```

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
                                seed=42)
# Run and extract results...
```

## Setup

Clone the repository:
```
git clone https://github.com/harrisonfloam/march-madness.git
```

Activate the development environment:
```
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