# Model_Util.py
"""Utility for resolving the AI model ID."""

from openai import OpenAI


def resolve_model(openai_client: OpenAI, env_model_id: str | None) -> str:
    """
    Resolve the model ID to use for AI operations.

    Args:
        openai_client: The OpenAI client instance (unused, for future model listing).
        env_model_id: The model ID from environment, or None.

    Returns:
        The resolved model ID, defaulting to 'typhoon-v1.5' if not specified.
    """
    return env_model_id or "typhoon-v1.5"