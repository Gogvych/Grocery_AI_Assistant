from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models import BaseLanguageModel


def create_budgeting_agent(llm: BaseLanguageModel):
    prompt = PromptTemplate.from_template(
        """
        You are a Budgeting Agent. Check if the provided product list fits within the given budget.
        Products: {product_list_raw}
        Budget: ${budget}
        Return a clear statement whether it fits the budget, and if not, suggest adjustments.
        """
    )
    return prompt | llm | RunnableLambda(lambda x: (print(f"Budgeting output: {x.content}"), {"budget_check_result": x.content})[1])