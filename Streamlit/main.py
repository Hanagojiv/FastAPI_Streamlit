import streamlit as st
import requests

# Streamlit UI
st.title("SEC Forms Q&A")

# Input for user question
user_question = st.text_input("Ask your question:")

# Button to submit the question
if st.button("Get Answer"):
    if not user_question:
        st.warning("Please enter a question.")
    else:
        # Send the question to the FastAPI service
        response = requests.post("http://localhost:8000/ask", json={"query": user_question})
        
        if response.status_code == 200:
            answer = response.json()["answer"]
            st.subheader("Answer:")
            st.write(answer)
        else:
            st.error("Error: Failed to retrieve an answer.")

