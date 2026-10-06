"""Central configuration for models and retrieval parameters."""

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

DEFAULT_K = 3

# Initial MMR experiment parameters. Not tuned: no sweep over fetch_k or
# lambda_mult was run.
MMR_FETCH_K = 10
MMR_LAMBDA_MULT = 0.5
