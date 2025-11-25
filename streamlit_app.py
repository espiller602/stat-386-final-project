"""
Streamlit app for NFL Playoff Predictor.

This app provides an interface for exploring NFL playoff passing statistics
and making predictions about postseason wins.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our package functions
from nfl_playoff_predictor.wrangling import (
    build_advanced_dataset,
    clean_column_names,
    process_and_save_dataset
)

# Page configuration
st.set_page_config(
    page_title="NFL Playoff Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set style
sns.set_style("whitegrid")
plt.style.use("seaborn-v0_8")


@st.cache_data
def load_data():
    """
    Load the cleaned playoff passing dataset.
    
    Returns
    -------
    pd.DataFrame
        Cleaned dataset with playoff passing statistics
    """
    data_file = Path("clean_playoff_passing.csv")
    
    if data_file.exists():
        df = pd.read_csv(data_file)
        return df
    else:
        # If file doesn't exist, try to create it
        data_dir = Path("data sources")
        if data_dir.exists():
            df = process_and_save_dataset(
                str(data_dir),
                str(data_file),
                start_year=2018,
                end_year=2024
            )
            return df
        else:
            st.error("Data file not found. Please ensure 'clean_playoff_passing.csv' exists or 'data sources' directory is available.")
            return None


def home_page():
    """Display the home/about page."""
    st.title("🏈 NFL Playoff Predictor")
    st.markdown("---")
    
    st.header("About")
    st.markdown("""
    This application predicts NFL postseason wins using advanced passing statistics 
    from Pro-Football-Reference. The project analyzes quarterback performance metrics 
    including:
    
    - **Completed Air Yards (CAY)** and **CAY per Attempt**
    - **Yards After Catch (YAC)** and **YAC per Completion**
    - **Intended Air Yards per Attempt (IAY/PA)**
    - **Pressure percentage**, **Drop rate**, and other advanced metrics
    
    ### Dataset
    The dataset includes NFL playoff passing statistics from **2018-2024**, focusing on 
    the period for which advanced statistics are reliably available.
    
    ### Project Information
    - **Course**: STAT 386
    - **Team**: Eli Spiller, Zion Tippetts
    - **Date**: November 2025
    """)
    
    st.markdown("---")
    
    st.header("Navigation")
    st.markdown("""
    Use the sidebar to navigate between pages:
    
    - **🏠 Home**: This page with project information
    - **📊 Data Explorer**: Explore and visualize the dataset
    - **🔮 Predictions**: Make predictions about playoff wins (coming soon)
    """)


def data_explorer_page(df):
    """Display the data exploration page."""
    st.title("📊 Data Explorer")
    st.markdown("---")
    
    if df is None:
        st.warning("No data available. Please check data files.")
        return
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Season filter
    seasons = sorted(df['Season'].unique()) if 'Season' in df.columns else []
    selected_seasons = st.sidebar.multiselect(
        "Select Seasons",
        options=seasons,
        default=seasons
    )
    
    # Team filter
    teams = sorted(df['Team'].unique()) if 'Team' in df.columns else []
    selected_teams = st.sidebar.multiselect(
        "Select Teams",
        options=teams,
        default=[]
    )
    
    # Apply filters
    filtered_df = df.copy()
    if selected_seasons:
        filtered_df = filtered_df[filtered_df['Season'].isin(selected_seasons)]
    if selected_teams:
        filtered_df = filtered_df[filtered_df['Team'].isin(selected_teams)]
    
    # Display dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(filtered_df))
    with col2:
        st.metric("Unique Players", filtered_df['Player'].nunique() if 'Player' in filtered_df.columns else 0)
    with col3:
        st.metric("Unique Teams", filtered_df['Team'].nunique() if 'Team' in filtered_df.columns else 0)
    with col4:
        st.metric("Seasons", filtered_df['Season'].nunique() if 'Season' in filtered_df.columns else 0)
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Table", "📈 Statistics", "📊 Visualizations", "🔍 Column Info"])
    
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name="filtered_playoff_passing.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.subheader("Descriptive Statistics")
        
        # Select columns for statistics
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_stats_cols = st.multiselect(
                "Select columns for statistics",
                options=numeric_cols,
                default=numeric_cols[:10] if len(numeric_cols) > 10 else numeric_cols
            )
            
            if selected_stats_cols:
                st.dataframe(
                    filtered_df[selected_stats_cols].describe(),
                    use_container_width=True
                )
        else:
            st.info("No numeric columns found for statistics.")
    
    with tab3:
        st.subheader("Data Visualizations")
        
        if len(numeric_cols) > 0:
            # Histogram
            st.write("#### Distribution of Playoff Games Won")
            if 'playoff_games_won' in filtered_df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                filtered_df['playoff_games_won'].hist(bins=20, ax=ax, edgecolor='black')
                ax.set_xlabel("Playoff Games Won")
                ax.set_ylabel("Frequency")
                ax.set_title("Distribution of Playoff Games Won")
                st.pyplot(fig)
                plt.close()
            
            # Scatter plot options
            st.write("#### Scatter Plot")
            col_x = st.selectbox("X-axis", options=numeric_cols, key="scatter_x")
            col_y = st.selectbox("Y-axis", options=numeric_cols, key="scatter_y")
            
            if col_x and col_y:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(filtered_df[col_x], filtered_df[col_y], alpha=0.6)
                ax.set_xlabel(col_x)
                ax.set_ylabel(col_y)
                ax.set_title(f"{col_y} vs {col_x}")
                st.pyplot(fig)
                plt.close()
            
            # Correlation heatmap
            st.write("#### Correlation Heatmap")
            if len(numeric_cols) > 1:
                corr_cols = st.multiselect(
                    "Select columns for correlation",
                    options=numeric_cols,
                    default=numeric_cols[:15] if len(numeric_cols) > 15 else numeric_cols
                )
                
                if len(corr_cols) > 1:
                    fig, ax = plt.subplots(figsize=(12, 10))
                    corr_matrix = filtered_df[corr_cols].corr()
                    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                               center=0, square=True, ax=ax)
                    ax.set_title("Correlation Heatmap")
                    st.pyplot(fig)
                    plt.close()
        else:
            st.info("No numeric columns available for visualization.")
    
    with tab4:
        st.subheader("Column Information")
        st.write(f"**Total Columns**: {len(filtered_df.columns)}")
        st.write(f"**Total Rows**: {len(filtered_df)}")
        
        col_info = pd.DataFrame({
            'Column': filtered_df.columns,
            'Data Type': filtered_df.dtypes.astype(str),
            'Non-Null Count': filtered_df.count().values,
            'Null Count': filtered_df.isnull().sum().values
        })
        st.dataframe(col_info, use_container_width=True)


def predictions_page(df):
    """Display the predictions page (placeholder for model)."""
    st.title("🔮 Predictions")
    st.markdown("---")
    
    if df is None:
        st.warning("No data available. Please check data files.")
        return
    
    st.info("""
    ⚠️ **Model Integration Pending**
    
    This section will be populated with the predictive model once it's ready.
    The model will use advanced passing statistics to predict playoff wins.
    """)
    
    st.markdown("---")
    
    # Placeholder for input form
    st.subheader("Input Parameters")
    st.markdown("Enter quarterback statistics to predict playoff wins:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Advanced Passing Metrics**")
        iay_pa = st.number_input("IAY/PA (Intended Air Yards per Attempt)", 
                                 min_value=0.0, max_value=20.0, value=7.0, step=0.1)
        cay_pa = st.number_input("CAY/PA (Completed Air Yards per Attempt)", 
                                 min_value=0.0, max_value=15.0, value=4.0, step=0.1)
        yac_cmp = st.number_input("YAC/Cmp (Yards After Catch per Completion)", 
                                  min_value=0.0, max_value=10.0, value=5.0, step=0.1)
        prss_pct = st.number_input("Prss% (Pressure Percentage)", 
                                   min_value=0.0, max_value=100.0, value=20.0, step=0.1)
    
    with col2:
        st.write("**Additional Metrics**")
        pkt_time = st.number_input("PktTime (Pocket Time)", 
                                   min_value=0.0, max_value=5.0, value=2.5, step=0.1)
        drop_pct = st.number_input("Drop% (Drop Percentage)", 
                                   min_value=0.0, max_value=20.0, value=5.0, step=0.1)
        bad_pct = st.number_input("Bad% (Bad Throw Percentage)", 
                                 min_value=0.0, max_value=50.0, value=15.0, step=0.1)
        interceptions = st.number_input("Int (Interceptions)", 
                                       min_value=0, max_value=10, value=1, step=1)
    
    st.markdown("---")
    
    # Placeholder for prediction button and results
    if st.button("🔮 Predict Playoff Wins", type="primary", use_container_width=True):
        st.warning("""
        **Model not yet integrated**
        
        This feature will be available once the predictive model is implemented.
        The model will use the input parameters above to predict the number of 
        playoff games won.
        
        Expected output:
        - Predicted playoff wins
        - Confidence interval
        - Model performance metrics
        """)
    
    st.markdown("---")
    
    # Placeholder section for model information
    with st.expander("📚 Model Information (Coming Soon)"):
        st.markdown("""
        Once the model is integrated, this section will display:
        
        - **Model Type**: e.g., Logistic Regression, Decision Trees, etc.
        - **Model Performance**: Accuracy, Precision, Recall, ROC-AUC scores
        - **Feature Importance**: Which statistics are most predictive
        - **Model Assumptions**: Limitations and considerations
        """)


def main():
    """Main application function."""
    # Sidebar navigation
    st.sidebar.title("🏈 Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["🏠 Home", "📊 Data Explorer", "🔮 Predictions"],
        label_visibility="collapsed"
    )
    
    # Load data (cached)
    df = load_data()
    
    # Route to appropriate page
    if page == "🏠 Home":
        home_page()
    elif page == "📊 Data Explorer":
        data_explorer_page(df)
    elif page == "🔮 Predictions":
        predictions_page(df)


if __name__ == "__main__":
    main()

