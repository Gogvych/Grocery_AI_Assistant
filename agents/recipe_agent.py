# agents/recipe_agent.py
from typing import Dict, Any
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.language_models import BaseLanguageModel

class RecipeAgent:
    def __init__(self, model: BaseLanguageModel):
        self.model = model
        #recipe cache
        self.cache = {}
        self.prompt = PromptTemplate.from_template(
            "Find a recipe for this request: {user_input}."
        )
        self.chain: Runnable = self.prompt | self.model

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        raw_response = inputs.get("response", "")
        query = ""
        try:
            query = raw_response.content.strip().lower()
        except AttributeError:
            query = str(raw_response).strip().lower()

        print(f"Recipe agent input (from planner): {query}")
        if query in self.cache:
            recipe = self.cache[query]
        else:
            result = self.chain.invoke({"user_input": query})
            recipe = ""
            try:
                recipe = result.content
            except AttributeError:
                recipe = result.get("text", result) if isinstance(result, dict) else result

            print(f"Recipe agent output (raw from LLM): {recipe}")
            self.cache[query] = recipe
        return {"recipe_output": recipe}
