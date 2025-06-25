import pytest
from unittest.mock import MagicMock
from graph import GroceryState, recipe, find_products, check_budget, finalize_list, create_graph, \
    llm, planner, recipe_agent, product_finder, budgeting, finalizer # Import module-level instances

# Mock LLM response
class MockLLM:
    def invoke(self, *args, **kwargs):
        return "mocked LLM response"

@pytest.fixture(autouse=True)
def mock_agents(monkeypatch):
    mock_llm_instance = MockLLM()

    # Create specific mocks for each agent's expected return type/structure
    mock_planner_instance = MagicMock()
    # Planner agent is likely returning a string, which maps to 'plan' in GroceryState
    mock_planner_instance.return_value = {"plan": "mocked plan content"}

    mock_recipe_agent_instance = MagicMock()
    # RecipeAgent is called directly, and its output is merged into GroceryState
    mock_recipe_agent_instance.return_value = {"recipe_output": "mocked recipe output content"}

    mock_product_finder_instance = MagicMock()
    mock_product_finder_instance.invoke.return_value = {"product_list_raw": "mocked product list content"}

    mock_budgeting_instance = MagicMock()
    mock_budgeting_instance.invoke.return_value = {"budget_check_result": "mocked budget check content"}

    mock_finalizer_instance = MagicMock()
    mock_finalizer_instance.invoke.return_value = {"final_shopping_list": "mocked final list content"}

    # Patch the module-level variables directly in graph.py
    monkeypatch.setattr("graph.llm", mock_llm_instance)
    monkeypatch.setattr("graph.planner", mock_planner_instance)
    monkeypatch.setattr("graph.recipe_agent", mock_recipe_agent_instance)
    monkeypatch.setattr("graph.product_finder", mock_product_finder_instance)
    monkeypatch.setattr("graph.budgeting", mock_budgeting_instance)
    monkeypatch.setattr("graph.finalizer", mock_finalizer_instance)


# Unit tests for individual node functions
def test_recipe_node():
    # recipe function expects inputs["plan"]
    inputs = GroceryState(user_input="test user input", plan="test plan content", recipe_output="", product_list_raw="", budget_check_result="", final_shopping_list="", budget=0.0)
    result = recipe(inputs)
    assert result == {"recipe_output": "mocked recipe output content"}

def test_find_products_node():
    inputs = GroceryState(user_input="", plan="", recipe_output="test recipe output content", product_list_raw="", budget_check_result="", final_shopping_list="", budget=0.0)
    result = find_products(inputs)
    assert result == {"product_list_raw": "mocked product list content"}

def test_check_budget_node():
    inputs = GroceryState(user_input="", plan="", recipe_output="", product_list_raw="test product list content", budget_check_result="", final_shopping_list="", budget=100.0)
    result = check_budget(inputs)
    assert result == {"budget_check_result": "mocked budget check content"}

def test_finalize_list_node():
    inputs = GroceryState(
        user_input="",
        plan="",
        recipe_output="test recipe output content",
        product_list_raw="test product list content",
        budget_check_result="test budget check result content",
        final_shopping_list="",
        budget=0.0
    )
    result = finalize_list(inputs)
    assert result == {"final_shopping_list": "mocked final list content"}

# Integration test for the entire graph
def test_create_graph_workflow():
    workflow = create_graph()
    
    # Define a sample input with a budget
    initial_state = {"user_input": "I need a recipe for chicken", "budget": 50.0}
    
    # Execute the workflow
    final_state = workflow.invoke(initial_state)

    # Assert that all relevant fields in the final state are populated by the mocks
    assert final_state["plan"] == "mocked plan content"
    assert final_state["recipe_output"] == "mocked recipe output content"
    assert final_state["product_list_raw"] == "mocked product list content"
    assert final_state["budget_check_result"] == "mocked budget check content"
    assert final_state["final_shopping_list"] == "mocked final list content"
    assert final_state["budget"] == 50.0 # Ensure budget is passed through 