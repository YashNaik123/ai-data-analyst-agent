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

    if st.session_state.controller.state.get("cleaning_report") is not None:
        with st.expander("See what cleaning was performed"):
            report = st.session_state.controller.state["cleaning_report"]
            if report:
                for note in report:
                    st.write("•", note)
            else:
                st.write("No cleaning was necessary.")

    st.write("---")
    question = st.text_input("Ask a business question about this data")

    if question and st.button("Analyze"):
        with st.spinner("Selecting columns and running analysis..."):
            result = st.session_state.controller.ask(question)

        if "error" in result:
            st.error(result["error"])
            if result.get("attempted_columns") is not None:
                st.write("Columns it tried to use:", result["attempted_columns"])
            st.info("Check your terminal for `[column_selector]` debug output showing what the model actually returned.")
        else:
            st.write("**Method used:**", result["route"]["method"], "—", result["route"]["rationale"])
            st.write("**Columns auto-selected:**", result["auto_selected_columns"])

            chart = result["chart"].get("chart")
            if chart is not None:
                st.plotly_chart(chart)
            elif result["chart"].get("render_error"):
                st.warning(f"Chart could not be rendered: {result['chart']['render_error']}")
            st.write("**Chart rationale:**", result["chart"]["rationale"])

            st.write("**Observation:**", result["insight"].get("observation"))
            st.write("**Interpretation:**", result["insight"].get("interpretation"))

            if not result["insight"].get("verified", True):
                st.warning(f"Unverified numbers flagged: {result['insight'].get('unverified_numbers')}")