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

        elif "gallery" in result:
            st.write(f"**Generating individual charts for every attribute** (target for relationship charts: `{result['gallery_target']}`)")
            for chart_item in result["gallery"]:
                st.write(f"**{chart_item['title']}**")
                st.plotly_chart(chart_item["figure"], key=chart_item["title"])

        else:
            st.write("**Method used:**", result["route"]["method"], "—", result["route"]["rationale"])
            st.write("**Columns auto-selected:**", result["auto_selected_columns"])

            skipped = result["analysis"].get("skipped_columns")
            if skipped:
                st.info(f"Columns skipped (not usable for this analysis): {skipped}")

            chart = result["chart"].get("chart")
            if chart is not None:
                st.plotly_chart(chart)
            elif result["chart"].get("render_error"):
                st.warning(f"Chart could not be rendered: {result['chart']['render_error']}")
            st.write("**Chart rationale:**", result["chart"]["rationale"])

            st.write("**Observation:**", result["insight"].get("observation"))
            st.write("**Interpretation:**", result["insight"].get("interpretation"))
            st.write("**Recommendation:**", result["insight"].get("recommendation"))

            if not result["insight"].get("coverage_complete", True):
                st.warning(f"Note: the insight may have skipped these attributes: {result['insight'].get('missing_attributes')}")

            if not result["insight"].get("verified", True):
                st.warning(f"Unverified numbers flagged: {result['insight'].get('unverified_numbers')}")