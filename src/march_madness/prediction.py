"""Tournament prediction classes and methods."""

from typing import TYPE_CHECKING
import numpy as np
import json
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from march_madness.tournament import Tournament, Game
from march_madness.types import Prediction

class LLMPredictionResponse(BaseModel):
    winner: int = Field(..., description="The winning team's number (1 or 2).", gt=0, lt=2, strict=True)
    confidence: float = Field(..., description="Confidence score (between 0 and 1).", ge=0.0, le=1.0)
    reasoning: str = Field(..., description="A concise explanation (max 2 sentences).")

def random_prediction(tournament: "Tournament", game: "Game", rng: np.random.Generator) -> Prediction:
    """Randomly picks one of the teams as the winner."""
    winner = str(rng.choice([game.team1, game.team2]))
    return winner, {"confidence": 0.50}

def team1_always_wins(tournament: "Tournament", game: "Game", rng: np.random.Generator) -> Prediction:
    """Always selects team1 as the winner."""
    return game.team1, None

def llm_prediction(tournament: "Tournament", game: "Game", rng: np.random.Generator, model_name: str) -> Prediction:
    """Use an LLM to select a winner."""
    seed = rng.integers(1, 2**32)
    
    # OpenAI models
    if "gpt" in model_name.lower():
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if OPENAI_API_KEY:
            llm = ChatOpenAI(
                model=model_name,
                temperature=0.8,
                seed=seed,
            )
            llm = llm.bind(response_format={"type": "json_object"})
            llm = llm.with_structured_output(LLMPredictionResponse)
        else:
            raise ValueError("OPENAI_API_KEY not found. Specify one in .env.")
    # Ollama models
    elif model_name in ["mistral", "llama3.2:1b"]:
        llm = ChatOllama(
            model = model_name, 
            temperature = 0.8,
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
            - confidence: A confidence rating for your prediction.
            - reasoning: A concise explanation of your reasoning, max 2 sentences.
            
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
            """
        ))
    ])

    formatted_prompt = prompt_template.format(
        team1=game.team1,
        team2=game.team2,
        team1_seed=game.team1_details["seed"],
        team2_seed=game.team2_details["seed"],
        round=game.round_number,
        tournament_state="Unavailable",
        historical_context="Unavailable"
    )

    response = llm.invoke(formatted_prompt)
    parsed_response = response
    
    return (
        getattr(game, f"team{parsed_response.winner}"),
        {"confidence": parsed_response.confidence, "reasoning": parsed_response.reasoning}
    )
    
"""
            Respond **only** in this format as a valid JSON object:
            {{
                "winner": "Team Name",
                "confidence": 0.50,
                "reasoning": "Concise but comprehensive explanation (2 sentences max)."
            }}
            
            Example:
            ```
            {{
                "winner": "Duke",
                "confidence": 0.75,
                "reasoning": "Duke has a better record against top 10 teams and a stronger defensive efficiency."
            }}
"""