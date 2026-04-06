import dspy
import os
from dotenv import load_dotenv

load_dotenv()


def configure(check_lm=False):
    """Configure LM from .env"""
    model_id = os.environ["LLM_MODEL"]
    api_base = os.environ["LLM_API_BASE"]
    api_key = os.environ["LLM_API_KEY"]
    lm = dspy.LM(model=model_id, api_base=api_base, api_key=api_key)
    if check_lm:
        response = lm("Hello?")
        print(response)
        print("")
    dspy.configure(lm=lm)


if __name__ == "__main__":
    configure(True)
