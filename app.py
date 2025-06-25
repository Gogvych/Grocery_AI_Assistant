import streamlit as st
from Supervisor import supervisor
import re



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
        # Use supervisor.stream to see intermediate steps for debugging
        full_response = []
        for chunk in supervisor.stream(inputs):
            full_response.append(chunk)
            st.write("**Intermediate Output:**", chunk)
            
        # Process the final result
        result = full_response[-1] if full_response else {}
        st.subheader("Final Shopping List")
        st.write(result.get("final_shopping_list", "No list generated."))