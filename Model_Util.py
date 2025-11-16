# Chatgpt ทำอันนี้ไปทำไม
def resolve_model(openai_client, env_model_id: str | None) -> str:
    return env_model_id or "typhoon-v1.5"