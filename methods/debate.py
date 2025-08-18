from openai import OpenAI, AzureOpenAI
from src.agents import Agent, Dialogue, Conversation

def debate(
    question: str, 
    solver_client: OpenAI | AzureOpenAI,
    solver_model: str,
    critic_client: AzureOpenAI,
    critic_model: str,
    max_rounds: int = 5
) -> Conversation:
    """Generates an answer through a debate between two agents."""
    critic_prompt = (
        "You are the Critic. Question assumptions and point out flaws, but do not solve it for them. "
        "If they have arrived at the correct answer, end the conversation immediately by saying TERMINATE <answer>."
    )
    solver_prompt = (
        "You are a helpful assistant. Think step by step. Incorporate feedback if you are wrong. Answer format: {\"answer\": <number>}."
    )

    critic = Agent(name="Critic", system_message=critic_prompt, model_name=critic_model, client=critic_client, temperature=0.3)
    solver = Agent(name="Solver", system_message=solver_prompt, model_name=solver_model, client=solver_client, temperature=0.7)

    dialogue = Dialogue(critic, solver)
    return dialogue.run(question, max_turns=max_rounds * 2)