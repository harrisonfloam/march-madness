"""Tournament prediction classes and methods."""

from typing import TYPE_CHECKING
import numpy as np
import json
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

if TYPE_CHECKING:
    from march_madness.tournament import Tournament, Game
from march_madness.types import Prediction


def random_prediction(tournament: "Tournament", game: "Game", rng: np.random.Generator,
                      p_team1: float = 0.50) -> Prediction:
    """Randomly picks one of the teams as the winner."""
    winner = str(rng.choice([game.team1, game.team2], p=[p_team1, 1-p_team1]))
    confidence = p_team1 if winner == game.team1 else (1 - p_team1)
    prediction_details = {
        "confidence": confidence,
        "reasoning": f"Randomly selected {winner} as the winner."
    }
    return winner, prediction_details

def team1_always_wins(tournament: "Tournament", game: "Game", rng: np.random.Generator) -> Prediction:
    """Always selects team1 as the winner."""
    prediction_details = {
        "confidence": 1.0, 
        "reasoning": "Team 1 always wins."
    }
    return game.team1, prediction_details

def llm_prediction(tournament: "Tournament", game: "Game", rng: np.random.Generator, 
                   model_name: str) -> Prediction:
    """Use an LLM to select a winner."""
    seed = rng.integers(1, 2**32)   # Generate a new seed so that each prediction isn't seeded the same
    
    # OpenAI models
    if "gpt" in model_name.lower():
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if OPENAI_API_KEY:
            # ChatClass = ChatOpenAI
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.6,
                seed=seed,
            )
            llm = llm.bind(response_format={"type": "json_object"})
            llm = llm.with_structured_output(LLMPredictionResponse)
        else:
            raise ValueError("OPENAI_API_KEY not found. Specify one in .env.")
    # Ollama models
    elif model_name in ["mistral", "llama3.2:1b"]:
        # ChatClass = ChatOllama
        llm = ChatOllama(
            model = model_name, 
            temperature = 0.6,
            seed=seed,
            format=LLMPredictionResponse.model_json_schema()
        )
        llm = llm.with_structured_output(LLMPredictionResponse, method="json_schema")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", (
            """
            You are a NCAA college basketball expert. You can make predictions regarding the outcome of future games and accurately state your confidence level and reasoning. The 2025 NCAA March Madness tournament is beginning soon nd you are tasked with predicting its outcomes.

            When evaluating a certain game, thoroughly consider these key factors (and any other facts you deem relevant):
            - Historical Matchups & Tournament Trends: Review past meetings, tournament performances, and trends for similar teams.  
            - Team Details as of DATE: Evaluate rosters, injuries, coaching strategies, and key player changes from previous seasons.  
            - Tournament Performance & Momentum: Assess how teams have played in earlier rounds, including upsets, fatigue, and adaptability.  
            - Regular Season Performance & Strength of Schedule: Consider key wins, conference difficulty, and performance against top opponents.  
            - Tactical Matchups & Game Dynamics: Analyze team strengths, weaknesses, player matchups, and coaching strategies under pressure.
            
            Response **only** with a valid `LLMPredictionResponse` Pydantic object with the following fields populated as specified:
            - winner: The number of the winning team (1 or 2) as specified in the prompt.
            - confidence: The probability of your prediction being correct (float value between 0 and 1).
            - reasoning: A concise explanation of your reasoning, using the actual team names, not numbers (max 2 sentences, single line, plaintext).
            
            The matchup to be predicted will be specified in this format:
                Team 1: '{{team1}}', seed {{team1_seed}}
                Team 2: '{{team2}}', seed {{team2_seed}}
                Round: {{round}}
                
            Notes:
            - Every prediction *must* select a winning team from the given matchup, no matter how low your confidence level is.
            - Your response must *always* be in the specified format and include non-null values for all specified keys.
            - Although you must respond with a team number, you should reason about the teams using their full names and map them back to their number once you reach a conclusion.
            """
        )),
        ("human", (
            """
            Current matchup:
                Team 1: '{team1}', seed {team1_seed}
                Team 2: '{team2}', seed {team2_seed}
                Round: {round}
            
            Current tournament state:
            '''
            {tournament_state}
            '''
            
            Historical context:
            '''
            {historical_context}
            '''
            
            If any errors have been encountered validating your response to this same prompt, they will appear below. Correct your response to resolve them, if any.
            '''
            {error_context}
            '''
            """
        ))
    ])
    tournament_state = "\n".join(str(game) for game in tournament.get_tournament_state())

    #NOTE: retries shouldnt be needed since validation fails silently
    MAX_RETRIES = 5
    error_messages = []
    for attempt in range(MAX_RETRIES):
        try:
            error_context = "\n".join(f"Error {i}: {msg}" for i, msg in reversed(list(enumerate(error_messages, start=1)))) if error_messages else ""
            formatted_prompt = prompt_template.format(
                team1=game.team1,
                team2=game.team2,
                team1_seed=game.team1_details["seed"],
                team2_seed=game.team2_details["seed"],
                round=game.round_number,
                tournament_state=tournament_state,
                historical_context="Unavailable",
                error_context=error_context
            )
            response = llm.invoke(formatted_prompt)
            
            winner = getattr(game, f"team{response.winner}")
            prediction_details = {
                "confidence": response.confidence, 
                "reasoning": response.reasoning
            }
            return winner, prediction_details

        except (ValidationError, OutputParserException) as e:
            error_messages.append(str(e))
            print(f"Warning: LLM output parsing failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")

    raise RuntimeError(f"LLM output parsing failed after {MAX_RETRIES} attempts.")


class LLMPredictionResponse(BaseModel):
    winner: int = Field(..., description="The winning team's number (1 or 2).")
    confidence: float = Field(..., description="The probability of the prediction being correct (between 0 and 1).")
    reasoning: str = Field(..., description="A concise explanation (max 2 sentences, single line).")
    
    # Store raw values
    raw_winner: int | None = None
    raw_confidence: float | None = None
    raw_reasoning: str | None = None
    
    @model_validator(mode="before")
    @classmethod
    def store_raw_values(cls, data):
        """Store raw input values before validation."""
        if isinstance(data, dict):
            return {
                **data,
                "raw_winner": data.get("winner"),
                "raw_confidence": data.get("confidence"),
                "raw_reasoning": data.get("reasoning"),
            }
        return data
    
    @field_validator("winner", mode="before")
    @classmethod
    def coerce_winner(cls, v):
        if not isinstance(v, int):
            try:
                v = int(v)
            except ValueError:
                return 1  # Default to 1...
        return 1 if v < 1.5 else 2  # Closest valid value

    @field_validator("confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v):
        # Convert to float
        if not isinstance(v, (int, float)):
            try:
                v = float(v)
            except ValueError:
                return 0.5  # Default
        # LLM used 0-10 scale
        if 1 < v <= 10:
            v /= 10
        # LLM used 0-100 scale
        elif 10 < v <= 100:
            v /= 100
        # Unknown scale
        elif v > 100:
            return 0.5  # Default
        # Negative probability
        elif v < 0:
            return 0.0
        return max(0, min(v, 1))
        
    @field_validator("reasoning", mode="before")
    @classmethod
    def coerce_reasoning(cls, v):
        if not isinstance(v, str):
            try:
                v = str(v)
            except ValueError:
                return "Unavailable."
        return v
    