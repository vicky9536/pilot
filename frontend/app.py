import streamlit as st
import requests

# Set FastAPI Backend URL
FASTAPI_URL = "http://localhost:8000"  # Change this if backend runs on a server

# Streamlit UI Layout
st.title("AI-Powered PDF Search & Q&A")

st.sidebar.header("Upload & Process PDFs")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])

# Upload PDF Function
if uploaded_file:
    st.sidebar.write("Uploading & Processing PDF...")
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(f"{FASTAPI_URL}/upload_pdf", files=files)

    if response.status_code == 200:
        st.sidebar.success("PDF processed successfully!")
    else:
        st.sidebar.error("Error processing PDF.")

# Semantic Search
st.header("Semantic Search")
query = st.text_input("Enter a search query:")
if st.button("Search"):
    if query:
        response = requests.post(f"{FASTAPI_URL}/search", json={"query": query})
        if response.status_code == 200:
            results = response.json().get("results", [])
            if results:
                st.write("Top Search Results:")
                for i, result in enumerate(results):
                    st.write(f"{i+1}. {result}")
            else:
                st.warning("No results found.")
        else:
            st.error("Error retrieving search results.")

# Question Answering
st.header("Ask a Question")
question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if question:
        response = requests.post(f"{FASTAPI_URL}/answer", json={"query": question})
        if response.status_code == 200:
            answer_data = response.json()
            answer = answer_data.get("answer", "No answer found.")
            sources = answer_data.get("sources", [])
            st.write("Answer:")
            st.write(answer)

            if sources:
                st.write("### Sources:")
                for src in sources:
                    st.write(f"- {src}")

        else:
            st.error("Error retrieving answer.")
