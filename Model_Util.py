# Model_Util.py
from openai import OpenAI

def resolve_model(client: OpenAI, env_model: str | None):
    if env_model:
        return env_model
    try:
        ms = client.models.list()
        ids = [m.id for m in ms.data]
        for cand in ids:
            low = cand.lower()
            if "typhoon" in low and ("instruct" in low or "chat" in low or "v" in low):
                return cand
        if ids:
            return ids[0]
    except Exception:
        pass
    return "typhoon-v1"