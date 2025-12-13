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
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Import our package functions
from nfl_playoff_predictor.wrangling import (
    build_advanced_dataset,
    clean_column_names,
    process_and_save_dataset
)
from nfl_playoff_predictor.analysis import (
    get_default_model,
    predict_playoff_wins,
    evaluate_model,
    prepare_data
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


@st.cache_resource
def load_model(df):
    """
    Load or train the predictive model.
    
    Parameters
    ----------
    df : pd.DataFrame
        Dataset for training
    
    Returns
    -------
    statsmodels.genmod.generalized_linear_model.GLMResults
        Trained model
    """
    if df is None:
        return None
    
    try:
        model = get_default_model(df)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
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
    - **Date**: Fall 2025
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
    if 'Season' in df.columns:
        # Convert to numeric, filter out NaN, then sort
        seasons = sorted([int(s) for s in df['Season'].unique() if pd.notna(s)])
    else:
        seasons = []
    selected_seasons = st.sidebar.multiselect(
        "Select Seasons",
        options=seasons,
        default=seasons
    )
    
    # Team filter
    if 'Team' in df.columns:
        # Convert Team column to string (handling NaN) for consistent filtering
        # Get unique teams, convert to string, filter out NaN, then sort
        teams = sorted([str(t) for t in df['Team'].unique() if pd.notna(t)])
    else:
        teams = []
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
        # Convert Team column to string for consistent type matching
        filtered_df = filtered_df[filtered_df['Team'].astype(str).isin(selected_teams)]
    
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
                # Set x-axis ticks to integers only
                max_games = int(filtered_df['playoff_games_won'].max())
                ax.set_xticks(range(0, max_games + 1))
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
            
            # Model Residual Plots
            st.write("#### Model Residual Diagnostics")
            try:
                # Get model for residual analysis (uses full dataset)
                model = load_model(df)
                if model is not None and hasattr(model, 'resid_pearson'):
                    # Get residuals and fitted values from the model
                    residuals = model.resid_pearson
                    fitted = model.fittedvalues
                    
                    # Residuals vs Fitted
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(fitted, residuals, alpha=0.7)
                    ax.axhline(0, color='red', linestyle='--')
                    ax.set_xlabel("Fitted values")
                    ax.set_ylabel("Pearson residuals")
                    ax.set_title("Residuals vs Fitted")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                    
                    # Histogram of residuals
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.hist(residuals, bins=15, density=True, alpha=0.6, edgecolor='black')
                    ax.set_xlabel("Pearson Residuals")
                    ax.set_ylabel("Density")
                    ax.set_title("Histogram of Pearson Residuals")
                    ax.grid(True, alpha=0.3)
                    st.pyplot(fig)
                    plt.close()
                    
                    # QQ Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sm.qqplot(residuals, line='45', ax=ax)
                    ax.set_title("QQ Plot of Pearson Residuals")
                    st.pyplot(fig)
                    plt.close()
                else:
                    st.warning("Model not available or does not have residual attributes.")
            except Exception as e:
                st.warning(f"Could not generate residual plots: {str(e)}")
            
            # Clustering Analysis
            st.write("#### Clustering Analysis")
            
            try:
                # Prepare data to rename columns (IAY/PA -> IAY_PA, etc.)
                df_prepared, _ = prepare_data(filtered_df.copy())
                
                # Map of feature names (both original and renamed)
                feature_map = {
                    'IAY_PA': ['IAY_PA', 'IAY/PA'],
                    'YAC_Cmp': ['YAC_Cmp', 'YAC/Cmp'],
                    'IntPerAtt': ['IntPerAtt'],
                    'DropPct': ['DropPct', 'Drop%'],
                    'BadPct': ['BadPct', 'Bad%']
                }
                
                # Check which features are available (check both original and renamed names)
                available_features = []
                feature_mapping = {}  # Map to standardized names
                
                for std_name, possible_names in feature_map.items():
                    found = False
                    for name in possible_names:
                        if name in df_prepared.columns:
                            available_features.append(std_name)
                            feature_mapping[std_name] = name
                            found = True
                            break
                    if not found and std_name == 'IntPerAtt':
                        # Try to calculate IntPerAtt
                        if 'Int' in df_prepared.columns and 'Att' in df_prepared.columns:
                            df_prepared['IntPerAtt'] = df_prepared['Int'] / df_prepared['Att']
                            available_features.append('IntPerAtt')
                            feature_mapping['IntPerAtt'] = 'IntPerAtt'
                
                if len(available_features) >= 3:
                    # Use standardized feature names, but get actual column names from mapping
                    actual_cols = [feature_mapping[f] for f in available_features]
                    # Prepare data for clustering (use actual column names)
                    clustering_data = df_prepared[actual_cols].dropna()
                    
                    if len(clustering_data) > 0:
                        # Standardize
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(clustering_data)
                        
                        # Elbow Method
                        st.write("##### Elbow Method for Optimal k")
                        inertia = []
                        K = range(1, 10)
                        for k in K:
                            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                            kmeans.fit(X_scaled)
                            inertia.append(kmeans.inertia_)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        ax.plot(K, inertia, 'bo-')
                        ax.set_xlabel('Number of clusters k')
                        ax.set_ylabel('Inertia')
                        ax.set_title('Elbow Method for k')
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                        
                        # KMeans Clustering with PCA
                        st.write("##### KMeans Clusters (k=2 and k=4)")
                        candidate_ks = [2, 4]
                        
                        # Prepare PCA
                        pca = PCA(n_components=2)
                        X_pca = pca.fit_transform(X_scaled)
                        
                        # Create side-by-side plots
                        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
                        
                        for i, k in enumerate(candidate_ks):
                            # Fit KMeans
                            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                            clusters = kmeans.fit_predict(X_scaled)
                            
                            # Plot
                            ax = axes[i]
                            for cluster in range(k):
                                subset = X_pca[clusters == cluster]
                                ax.scatter(subset[:, 0], subset[:, 1], label=f'Cluster {cluster}', s=50, alpha=0.6)
                            
                            ax.set_title(f'KMeans Clusters (k={k})')
                            ax.set_xlabel('PCA 1')
                            ax.set_ylabel('PCA 2')
                            ax.legend()
                            ax.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                        # Clusters with Feature Vectors
                        st.write("##### Clusters with Feature Vectors (k=2)")
                        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
                        clusters = kmeans.fit_predict(X_scaled)
                        
                        # Get loadings (use standardized feature names for display)
                        loadings = pd.DataFrame(pca.components_, columns=available_features, index=['PCA1', 'PCA2'])
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # Plot clusters
                        for cluster in range(2):
                            subset = X_pca[clusters == cluster]
                            ax.scatter(subset[:, 0], subset[:, 1], label=f'Cluster {cluster}', s=50, alpha=0.6)
                        
                        # Plot feature vectors
                        for i, feature in enumerate(available_features):
                            ax.arrow(0, 0, loadings.loc['PCA1', feature]*3, loadings.loc['PCA2', feature]*3, 
                                    color='r', alpha=0.7, head_width=0.1)
                            ax.text(loadings.loc['PCA1', feature]*3.2, loadings.loc['PCA2', feature]*3.2, 
                                   feature, color='r', fontsize=9)
                        
                        ax.set_xlabel('PCA 1')
                        ax.set_ylabel('PCA 2')
                        ax.set_title('Clusters with Feature Vectors')
                        ax.legend()
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close()
                    else:
                        st.warning("Not enough data points for clustering after removing missing values.")
                else:
                    # Show helpful message about what was found
                    found_cols = [c for c in df_prepared.columns if any(f in c for f in ['IAY', 'YAC', 'Int', 'Drop', 'Bad', 'Cmp'])]
                    st.info(f"Clustering analysis requires at least 3 of: IAY_PA (or IAY/PA), YAC_Cmp (or YAC/Cmp), IntPerAtt, DropPct (or Drop%), BadPct (or Bad%). Found {len(available_features)}: {available_features}. Available relevant columns: {found_cols[:10]}")
            except Exception as e:
                st.warning(f"Could not generate clustering visualizations: {str(e)}")
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
    """Display the predictions page with model integration."""
    st.title("🔮 Predictions")
    st.markdown("---")
    
    if df is None:
        st.warning("No data available. Please check data files.")
        return
    
    # Load model
    model = load_model(df)
    
    if model is None:
        st.error("Unable to load model. Please check the data and try again.")
        return
    
    # Display model information
    with st.expander("📚 Model Information", expanded=False):
        st.markdown("""
        **Model Type**: Poisson Generalized Linear Model (GLM)
        
        **Selected Variables** (from Cross-Validation, Rank 1):
        - IAY_PA (Intended Air Yards per Attempt)
        - YAC_Cmp (Yards After Catch per Completion)
        - IntPerAtt (Interceptions per Attempt = Int / Att)
        
        **Model Performance**:
        - Pseudo R-squared: ~0.15
        - Residual deviance / df: ~1.24
        - Selected via 5-fold cross-validation (best model by deviance)
        """)
    
    st.markdown("---")
    
    # Input form
    st.subheader("Input Parameters")
    st.markdown("Enter quarterback statistics to predict playoff wins:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Advanced Passing Metrics**")
        iay_pa = st.number_input("IAY/PA (Intended Air Yards per Attempt)", 
                                 min_value=0.0, max_value=20.0, value=7.0, step=0.1,
                                 help="Average intended air yards per pass attempt")
        yac_cmp = st.number_input("YAC/Cmp (Yards After Catch per Completion)", 
                                  min_value=0.0, max_value=10.0, value=5.0, step=0.1,
                                  help="Average yards after catch per completion")
    
    with col2:
        st.write("**Interception Metrics**")
        interceptions = st.number_input("Int (Total Interceptions)", 
                                       min_value=0, max_value=10, value=1, step=1,
                                       help="Total number of interceptions")
        attempts = st.number_input("Att (Total Pass Attempts)", 
                                  min_value=1, max_value=200, value=50, step=1,
                                  help="Total number of pass attempts")
        
        # Calculate IntPerAtt
        if attempts > 0:
            int_per_att = interceptions / attempts
            st.info(f"**IntPerAtt**: {int_per_att:.4f} (calculated as Int / Att)")
        else:
            int_per_att = 0.0
            st.warning("Attempts must be greater than 0")
    
    st.markdown("---")
    
    # Prediction button and results
    if st.button("🔮 Predict Playoff Wins", type="primary", use_container_width=True):
        try:
            # Calculate IntPerAtt
            if attempts <= 0:
                st.error("Pass attempts must be greater than 0")
                return
            
            int_per_att = interceptions / attempts
            
            # Prepare input dictionary with correct variable names (3 variables from CV)
            input_data = {
                'IAY_PA': iay_pa,
                'YAC_Cmp': yac_cmp,
                'IntPerAtt': int_per_att
            }
            
            # Make prediction
            prediction = predict_playoff_wins(model, input_data)
            
            # Display results
            st.success("### Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Playoff Wins", f"{prediction:.2f}")
            with col2:
                # Calculate confidence interval (approximate)
                lower_bound = max(0, prediction - 1.0)
                upper_bound = prediction + 1.0
                st.metric("Confidence Range", f"{lower_bound:.1f} - {upper_bound:.1f}")
            with col3:
                # Round to nearest integer for interpretation
                rounded_pred = round(prediction)
                st.metric("Rounded Prediction", f"{rounded_pred} games")
            
            st.info("""
            **Note**: This is a point prediction from a Poisson GLM model. 
            The actual number of playoff wins is a count variable (0, 1, 2, 3, ...).
            The model predicts the expected (mean) number of wins.
            """)
            
            # Show input summary
            with st.expander("📊 Input Summary"):
                st.write("**Input Parameters:**")
                st.write(f"- IAY_PA: {iay_pa}")
                st.write(f"- YAC_Cmp: {yac_cmp}")
                st.write(f"- Int: {interceptions}")
                st.write(f"- Att: {attempts}")
                st.write(f"- IntPerAtt: {int_per_att:.4f} (calculated)")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure all input values are within valid ranges.")
    
    st.markdown("---")
    
    # Model diagnostics section
    with st.expander("🔍 Model Diagnostics"):
        try:
            # Prepare data for evaluation
            df_prep, _ = prepare_data(df)
            
            # Calculate IntPerAtt for evaluation
            if 'Int' in df_prep.columns and 'Att' in df_prep.columns:
                df_prep['IntPerAtt'] = df_prep['Int'] / df_prep['Att']
            else:
                st.warning("Cannot calculate IntPerAtt - missing Int or Att columns")
                return
            
            # Use the correct selected variables (3 variables from CV)
            selected_vars = ['IAY_PA', 'YAC_Cmp', 'IntPerAtt']
            
            # Get evaluation metrics
            eval_results = evaluate_model(model, df_prep, selected_vars)
            
            st.write("**Model Evaluation Metrics:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pseudo R²", f"{eval_results['pseudo_r_squared']:.3f}")
            with col2:
                st.metric("AIC", f"{eval_results['aic']:.2f}")
            with col3:
                st.metric("Overdispersion", f"{eval_results['overdispersion_ratio']:.2f}")
            
            st.write("**Variance Inflation Factors (VIF):**")
            st.dataframe(eval_results['vif'], use_container_width=True)
            
            if eval_results['overdispersion_ratio'] > 1.5:
                st.warning("⚠️ Overdispersion detected. Consider using Negative Binomial regression.")
            else:
                st.success("✓ Overdispersion is within acceptable range for Poisson model.")
        
        except Exception as e:
            st.warning(f"Could not compute diagnostics: {str(e)}")


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

