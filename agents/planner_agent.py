from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models import BaseLanguageModel


def create_planner_agent(llm: BaseLanguageModel):
    prompt = PromptTemplate.from_template(
        """
        You are a Planner. Interpret the user's intent: {user_input}
        Return a clear plan with steps.
        """
    )
    return prompt | llm | RunnableLambda(lambda x: (print(f"Planner output: {x.content}"), {"plan": x.content})[1])