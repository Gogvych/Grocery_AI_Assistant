from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models import BaseLanguageModel


def create_product_finder_agent(llm: BaseLanguageModel):
    prompt = PromptTemplate.from_template(
        """
        You are a Product Finder. Map the ingredients from the provided recipe to products in a store.
        Recipe: {recipe_output}
        """
    )
    return prompt | llm | RunnableLambda(lambda x: (print(f"Product Finder output: {x.content}"), {"product_list_raw": x.content})[1])