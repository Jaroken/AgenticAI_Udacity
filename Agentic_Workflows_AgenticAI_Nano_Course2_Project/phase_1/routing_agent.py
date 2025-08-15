
# TODO: 1 - Import the KnowledgeAugmentedPromptAgent and RoutingAgent
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent

# Load environment variables from .env file
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor"

knowledge = "You know everything about Texas"
# Define the Texas Knowledge Augmented Prompt Agent
texas_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key,
                                            persona=persona,
                                            knowledge=knowledge)
knowledge = "You know everything about Europe"
# Define the Europe Knowledge Augmented Prompt Agent
europe_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key,
                                            persona=persona,
                                            knowledge=knowledge)
persona = "You are a college math professor"
knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
# Define the Math Knowledge Augmented Prompt Agent
math_agent = KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key,
                                           persona=persona,
                                           knowledge=knowledge)
routing_agent = RoutingAgent(openai_api_key, {})
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda prompt: texas_agent.respond(prompt)# Call the Texas Agent to respond to prompts
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda prompt: europe_agent.respond(prompt) # Define a function to call the Europe Agent
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda prompt: math_agent.respond(prompt) # Define a function to call the Math Agent
    }
]

routing_agent.agents = agents

# - Print the RoutingAgent responses to the following prompts:
#           - "Tell me about the history of Rome, Texas"
#           - "Tell me about the history of Rome, Italy"
#           - "One story takes 2 days, and there are 20 stories"

prompts = ["Tell me about the history of Rome, Texas",
           "Tell me about the history of Rome, Italy",
           "One story takes 2 days, and there are 20 stories"]

[print(routing_agent.route_prompts(prompt)) for prompt in prompts]