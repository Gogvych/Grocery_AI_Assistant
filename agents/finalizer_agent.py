from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models import BaseLanguageModel


def create_finalizer_agent(llm: BaseLanguageModel):
    prompt = PromptTemplate.from_template(
        """
        You are a Finalizer. Given the recipe, product list, and budget check result, create a final comprehensive shopping list.
        Recipe: {recipe_output}
        Products: {product_list_raw}
        Budget Check: {budget_check_result}
        Provide the final shopping list in a clear, itemized format.
        """
    )
    return prompt | llm | RunnableLambda(lambda x: (print(f"Finalizer output: {x.content}"), {"final_shopping_list": x.content})[1])