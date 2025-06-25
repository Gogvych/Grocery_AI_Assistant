import os
from typing import TypedDict
from dotenv import load_dotenv
load_dotenv()
# from huggingface_hub import login
# login()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langgraph.pregel import Pregel

from agents.planner import create_planner_tool
from agents.recipe import RecipeAgent, create_recipe_tool
from agents.product_finder import create_product_finder_tool
from agents.budgeting import create_budgeting_tool
from agents.finalizer import create_finalizer_tool

llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-Small-24B-Instruct-2501",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat = ChatHuggingFace(llm=llm, verbose=False)

# Initialize agents
planner_agent = create_react_agent(
    model=chat,
    tools=[create_planner_tool(llm)],
    prompt="You are a Planner. Use your tool to convert user intent into a structured plan."
)
budgeting_agent = create_react_agent(
    model=chat,
    tools=[create_planner_tool(llm)],
    prompt="You are a Budgeting Agent. Use your tool to check whether choosed items fit user's budget"
)
product_finder_agent = create_react_agent(
    model=chat,
    tools=[create_planner_tool(llm)],
    prompt="You are an Agent that finds products. Use your tool to map the ingredients from the provided recipe to products in a store"
)
recipe_agent = create_react_agent(
    model=chat,
    tools=[create_recipe_tool(RecipeAgent(llm))],
    prompt="You are a Recipe Agent. Use your tool to find recipes based on user requests."
)
finalizer_agent = create_react_agent(
    model=chat,
    tools=[create_planner_tool(llm)],
    prompt="You are a Finalizer. Given the recipe, product list, and budget check result, create a final comprehensive shopping list."
)

supervisor = create_supervisor(
    agents=[planner_agent, budgeting_agent, product_finder_agent, recipe_agent, finalizer_agent],
    model=chat,
    prompt=(
        """You are the Supervisor Agent, responsible for orchestrating the work of 
        five specialized agents to fulfill a user request. Your primary goals are to 
        manage the flow of execution, control data routing, 
        and ensure task completion in the correct order.

Task Flow:
Receive user input, such as:
"I want to buy ingredients for pizza." or "Prepare a shopping list for dinner for 4 people within a $25 budget"

Step 1 Planner Agent:
Send the user request to the Planner Agent.
Task: Identify user intent and break it down into a sequence of actionable steps.

Step 2 Recipe Agent:
Forward the goal (e.g., "make pizza") to the Recipe Agent.
Task: Find a suitable recipe, including ingredients and quantities.

Step 3 Product Finder Agent:
Send the list of ingredients to the Product Finder Agent.
Task: Match each ingredient to specific store products or product types.

Step 4 Budgeting Agent:
Provide the mapped products and a specified budget (if any) to the Budgeting Agent.
Task: Verify whether the shopping list is within budget and suggest alternatives if needed.

Step 5 Finalizer Agent:
Once budgeting is validated, pass the results to the Finalizer Agent.
Task: Compile the final shopping list, formatted for user presentation.

Supervisor Agent Responsibilities:
Maintain control over which agent is activated and when.
Route data outputs from one agent to the correct next agent.
Handle errors or missing data by rerouting or re-invoking prior agents as needed.
Return the final result to the user.

Example Instruction:
User Input: "I want to buy ingredients for pizza under $20."
Your Role: Coordinate the agents in sequence to produce a final budgeted shopping list for pizza ingredients.
"""   )
).compile()

# for chunk in supervisor.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
#             }
#         ]
#     }
# ):
#     print(chunk)
#     print("\n")


# messages = [
#     SystemMessage(content="You are a helpful assistant. "),
#     HumanMessage(content="What's the capital of France?"),
# ]

# response = chat.invoke(messages)
# print(response)
