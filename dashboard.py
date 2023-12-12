from utils.get_metrics import PaperAnalysis

import streamlit as st

# Assuming PaperAnalysis class is defined as above

# Initialize the app
paper_analysis = PaperAnalysis()

st.set_page_config(
    page_title="Corporatus",
    page_icon="📖",
    layout="wide",
)

placeholder = st.empty()

st.title("Real-Time Corporatus DB Metrics")

kpi1, kpi2, kpi3 = st.columns(3)
fig_col1, fig_col2= st.columns(2)
col1, col2 = st.columns(2)

# Refresh button
if st.button("Refresh Data"):
    papers_df = paper_analysis.get_papers_df()
    cluster_counts = paper_analysis.get_cluster_counts()
    all_pdfs = paper_analysis.get_total_papers()

    with placeholder.container():
        kpi1.metric(
            label="Total Papers processed ✅",
            value=len(papers_df)
        )

        kpi2.metric(
            label="Fields of knowledge 📚",
            value= 4
        )

        kpi3.metric(
            label="Papers Awaiting Processing ⏳",
            value=all_pdfs - len(papers_df)
        )

    with placeholder.container():
        with fig_col1:
            plt = paper_analysis.create_bar_chart()
            st.pyplot(plt)

        with fig_col2:
            plt = paper_analysis.visualize_clusters()
            st.pyplot(plt)

    with col1:
        st.metric(label="Health Sciences", value=paper_analysis.total_by_area['Health Sciences'])
        st.divider()
        st.metric(label="Social Sciences and Humanities", value=paper_analysis.total_by_area['Social Sciences and Humanities'])

    with col2:
        st.metric(label="Life Sciences", value=paper_analysis.total_by_area['Life Sciences'])
        st.divider()
        st.metric(label="Physical Sciences and Engineering", value=paper_analysis.total_by_area['Physical Sciences and Engineering'])

    #cluster_keywords_keybert = paper_analysis.get_summary()
