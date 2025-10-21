
import streamlit as st
import requests

st.set_page_config(page_title="FusionRAG Demo", layout="wide")
st.title("FusionRAG — Multimodal Product Intelligence (Demo)")

api_url = st.secrets.get("API_URL", "http://localhost:8000")

col1, col2 = st.columns(2)
with col1:
    text_query = st.text_input("Text Query", "red running shoes with heel support")
    if st.button("Search"):
        resp = requests.post(f"{api_url}/search", json={"text": text_query}).json()
        st.write(resp)

with col2:
    qa_query = st.text_input("Ask a Product Question", "Does it have breathable mesh?")
    if st.button("Ask"):
        resp = requests.post(f"{api_url}/qa", json={"text": qa_query}).json()
        st.write(resp)
