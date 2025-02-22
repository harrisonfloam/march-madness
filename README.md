# march-madness

A Python project for open-ended tournament management and simulation, specifically aimed at the NCAA March Madness tournament.

## Outline

- `src/march_madness/`: main project directory
  - tournament.py: 
  - predictor.py: 
- `notebooks/`: tutorial notebooks
  - march_madness.ipynb: 
- `data/`: .csv files containing tournament initialization data
  - march_madness_2024.csv

## Usage

### Tournament

TODO: params, attrs, methods, example

### TournamentSimulator

TODO: params, attrs, methods, example

### Game

TODO: params, attrs, methods, example

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