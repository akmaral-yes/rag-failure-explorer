"""Central configuration for models and retrieval parameters."""

import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

DEFAULT_K = 3

# MMR parameters are initial experiment parameters, not tuned constants — Phase 3
# validation sweeps fetch_k in {6, 10} and lambda_mult in {0.3, 0.5, 0.7} to confirm
# the diversity effect isn't an artifact of one arbitrary setting.
MMR_FETCH_K = 10
MMR_LAMBDA_MULT = 0.5
