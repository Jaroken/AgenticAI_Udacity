# Import the KnowledgeAugmentedPromptAgent class from workflow_agents
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# from project.starter.phase_1.action_planning_agent import knowledge
# from project.starter.phase_1.evaluation_agent import knowledge_agent

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
# - Instantiate a KnowledgeAugmentedPromptAgent with:
#           - Persona: "You are a college professor, your answer always starts with: Dear students,"
#           - Knowledge: "The capital of France is London, not Paris"
knowledge = "The capital of France is London, not Paris"

knowledge_agent= KnowledgeAugmentedPromptAgent(openai_api_key=openai_api_key,persona=persona, knowledge=knowledge)

# Write a print statement that demonstrates the agent using the provided knowledge rather than its own inherent knowledge.
response = knowledge_agent.respond(prompt)
print(response)
print("Please Note: The agent’s response explicitly uses the provided knowledge rather than its inherent knowledge "
      "inherent in the LLM.")