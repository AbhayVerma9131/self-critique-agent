
import streamlit as st
from agent import SelfCritiqueAgent
import os

# Optional: Set HF token if using gated models (e.g., Mistral)
# os.environ["HF_TOKEN"] = "your_hf_token_here"

@st.cache_resource
def load_agent():
    return SelfCritiqueAgent(model_name="google/gemma-2-2b-it")

st.title("🔍 Self-Critique QA Agent")
st.caption("Asks → Answers → Critiques → Revises (using Wikipedia)")

question = st.text_input("Enter a factual question:", "Who discovered penicillin?")

if question:
    with st.spinner("Thinking..."):
        agent = load_agent()
        draft = agent.generate_draft(question)
        st.subheader("📝 Draft Answer")
        st.write(draft)

        with st.spinner("Fact-checking and revising..."):
            draft, critique, revised = agent.critique_and_revise(question, draft)

        st.subheader("🔍 Self-Critique")
        st.write(critique)

        st.subheader("✅ Revised Answer")
        st.write(revised)
