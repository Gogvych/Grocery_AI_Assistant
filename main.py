import streamlit as st
from graph import create_graph
import re

graph = create_graph()

st.title("Grocery Shopping AI Assistant")
user_input = st.text_input("Enter your request:", "Prepare a shopping list for dinner for 4 people within a $25 budget")

# Extract budget from user_input using regex
budget_match = re.search(r'\$(\d+\.?\d*)', user_input)
if budget_match:
    try:
        budget_value = float(budget_match.group(1))
    except ValueError:
        st.warning("Could not parse budget from input. Using default budget of $25.")
        budget_value = 25.0
else:
    st.info("No budget specified in input. Using default budget of $25.")
    budget_value = 25.0

if st.button("Generate Shopping List"):
    with st.spinner("Processing..."):
        inputs = {"user_input": user_input, "budget": budget_value}
        result = graph.invoke(inputs)
        #Debugging purposes only
        st.subheader("Debug: Full Graph Result")
        st.write(result)
        
        final_list_content = result.get("final_shopping_list", "No list generated.")
        st.subheader("Debug: Type and Value of final_shopping_list")
        st.write(f"Type: {type(final_list_content)}")
        st.write(f"Value: {final_list_content}")
        
        st.subheader("Final Shopping List")
        st.write(final_list_content)