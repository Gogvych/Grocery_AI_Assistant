# graph.py
import os
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from huggingface_hub import InferenceClient
from langsmith.run_helpers import traceable
from langchain_groq import ChatGroq


#from custom_llm import HuggingFaceInferenceClientLLM

from agents.planner_agent import create_planner_agent
from agents.recipe_agent import RecipeAgent
from agents.product_finder_agent import create_product_finder_agent
from agents.budgeting_agent import create_budgeting_agent
from agents.finalizer_agent import create_finalizer_agent

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=512,
    timeout=None,
    max_retries=2
)


# Initialize agents
planner = create_planner_agent(llm)
recipe_agent = RecipeAgent(llm)
product_finder = create_product_finder_agent(llm)
budgeting = create_budgeting_agent(llm)
finalizer = create_finalizer_agent(llm)

class GroceryState(TypedDict):
    user_input: str
    plan: str
    recipe_output: str
    product_list_raw: str
    budget_check_result: str
    final_shopping_list: str
    budget: float

# Wrap recipe_agent in a callable function for LangGraph node
def recipe(inputs: GroceryState) -> dict:
    # The recipe agent expects 'user_input' and 'response' (from planner) as its input
    recipe_agent_input = {"user_input": inputs["user_input"], "response": inputs["plan"]}
    recipe_output_dict = recipe_agent(recipe_agent_input)
    return {"recipe_output": recipe_output_dict.get("recipe_output", "")}

def find_products(inputs: GroceryState) -> dict:
    product_finder_input = {"recipe_output": inputs["recipe_output"]}
    product_list_dict = product_finder.invoke(product_finder_input)
    return {"product_list_raw": product_list_dict.get("product_list_raw", "")}

def check_budget(inputs: GroceryState) -> dict:
    budgeting_input = {"product_list_raw": inputs["product_list_raw"], "budget": inputs["budget"]}
    budget_check_dict = budgeting.invoke(budgeting_input)
    return {"budget_check_result": budget_check_dict.get("budget_check_result", "")}

def finalize_list(inputs: GroceryState) -> dict:
    finalizer_input = {
        "recipe_output": inputs["recipe_output"],
        "product_list_raw": inputs["product_list_raw"],
        "budget_check_result": inputs["budget_check_result"]
    }
    final_list_dict = finalizer.invoke(finalizer_input)
    return {"final_shopping_list": final_list_dict.get("final_shopping_list", "")}

@traceable(name="SupervisorAgent")
def create_graph():
    workflow = StateGraph(state_schema=GroceryState)

    workflow.add_node("planner", planner)
    workflow.add_node("recipe", recipe)
    workflow.add_node("product_finder", find_products)
    workflow.add_node("budgeting", check_budget)
    workflow.add_node("finalizer", finalize_list)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "recipe")
    workflow.add_edge("recipe", "product_finder")
    workflow.add_edge("product_finder", "budgeting")
    workflow.add_edge("budgeting", "finalizer")
    workflow.add_edge("finalizer", END)

    return workflow.compile()
