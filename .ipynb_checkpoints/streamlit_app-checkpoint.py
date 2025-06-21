import streamlit as st
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded"
)
pages = {
    "Menu" : [
        st.Page("pages/Obesity_Analysis.py", title="Obesity Analysis",icon =":material/filter_alt:"),
        st.Page("pages/Malnutrition_Analysis.py", title="Malnutrition Analysis",icon =":material/filter_alt:"),
        st.Page("pages/querys.py", title="Queries",icon = ":material/list_alt:")
    ]
}
pg = st.navigation(pages)
pg.run()