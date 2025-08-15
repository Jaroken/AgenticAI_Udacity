# Test script for DirectPromptAgent class

# Import the DirectPromptAgent class from BaseAgents
import os
from dotenv import load_dotenv
from workflow_agents.base_agents import DirectPromptAgent

# Load environment variables from .env file
load_dotenv()

# Load the OpenAI API key from the environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

# Instantiate the DirectPromptAgent as direct_agent
direct_agent = DirectPromptAgent(openai_api_key=openai_api_key)
# Use direct_agent to send the prompt defined above and store the response
direct_agent_response = direct_agent.respond(prompt=prompt)

# Print the response from the agent
print(direct_agent_response)

# Print an explanatory message describing the knowledge source used by the agent to generate the response
print("Explanatory Message: This DirectPromptAgent agent uses the knowledge inherent in the chat gpt 3.5 turbo model. "
      "It does not use any persona or provided knowledge from an external source to influence its responses and "
      "relies on what the base model was trained on a number of years ago from current day 2025.")
