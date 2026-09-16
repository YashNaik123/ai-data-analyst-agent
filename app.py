import streamlit as st
from agents.controller import Controller

st.title("AI Data Analyst Agent")

if "controller" not in st.session_state:
    st.session_state.controller = Controller(backend="ollama")

uploaded_file = st.file_uploader("Upload a CSV", type="csv")

if uploaded_file:
    with open("temp.csv", "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "profile_summary" not in st.session_state or st.session_state.get("last_file") != uploaded_file.name:
        st.session_state.profile_summary = st.session_state.controller.load_data("temp.csv")
        st.session_state.last_file = uploaded_file.name

    st.write("**Data Profile Summary:**")
    st.write(st.session_state.profile_summary)

    st.write("---")
    question = st.text_input("Ask a business question about this data")

    df_columns = st.session_state.controller.state["df"].columns
    target = st.selectbox("Target column (what you're trying to explain)", df_columns)
    features = st.multiselect("Feature columns (what might explain it)", df_columns)

    if question and st.button("Analyze"):
        with st.spinner("Running analysis..."):
            result = st.session_state.controller.ask(
                question,
                method_kwargs={"target": target, "features": features}
            )

        if "error" in result:
            st.error(result["error"])
        else:
            st.write("**Method used:**", result["route"]["method"], "—", result["route"]["rationale"])

            if result["chart"]["chart"]:
                st.plotly_chart(result["chart"]["chart"])
            st.write("**Chart rationale:**", result["chart"]["rationale"])

            st.write("**Observation:**", result["insight"].get("observation"))
            st.write("**Interpretation:**", result["insight"].get("interpretation"))

            if not result["insight"].get("verified", True):
                st.warning(f"Unverified numbers flagged: {result['insight'].get('unverified_numbers')}")