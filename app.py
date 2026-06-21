import streamlit as st
import pandas as pd
import numpy as np
from recommender import ProductRecommender

# Set page configuration for a modern look
st.set_page_config(
    page_title="AI Product Recommendation System",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for custom styling, cards, and borders
st.markdown("""
<style>
    /* Header Style */
    .title-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .title-banner h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        color: white !important;
    }
    .title-banner p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Product Cards */
    .product-card {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid var(--primary-color);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    .product-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.12);
    }
    .product-name {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--primary-color);
        margin-bottom: 0.25rem;
    }
    .product-category {
        display: inline-block;
        background-color: rgba(128, 128, 128, 0.15);
        color: var(--text-color);
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .product-description {
        font-size: 0.95rem;
        color: var(--text-color);
        line-height: 1.4;
        opacity: 0.9;
    }
    
    /* Recommendation Card Specifics */
    .rec-card {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 1.25rem;
        transition: all 0.2s;
    }
    .rec-card:hover {
        border-color: var(--primary-color);
        box-shadow: 0 6px 18px rgba(0,0,0,0.1);
    }
    .rec-score-badge {
        float: right;
        background-color: #28a745;
        color: white !important;
        font-size: 0.85rem;
        font-weight: 700;
        padding: 0.3rem 0.75rem;
        border-radius: 8px;
    }
    
    /* Reason box styling */
    .reason-box {
        background-color: rgba(30, 144, 255, 0.08);
        border: 1px dashed rgba(30, 144, 255, 0.25);
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-top: 0.75rem;
        font-size: 0.9rem;
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# Helper to format percentages
def format_score(score):
    return f"{score * 100:.1f}%"

@st.cache_resource
def get_recommender():
    """Cache the recommender model initialization for fast reloading."""
    return ProductRecommender()

def main():
    # Load recommendation model
    try:
        recommender = get_recommender()
    except Exception as e:
        st.error(f"Failed to load dataset or initialize recommender. Did you run `generate_data.py` first? Error: {e}")
        return

    # Title Banner
    st.markdown("""
        <div class="title-banner">
            <h1>AI-Based Product Recommendation System</h1>
            <p>A Content-Based Filtering Engine powered by TF-IDF & Cosine Similarity (NLP)</p>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar setup
    st.sidebar.header("🔍 Recommendation Settings")
    st.sidebar.markdown("Use the controls below to select a target product and adjust the recommender parameters.")

    # 1. Sidebar Product Selection
    # Dropdown to filter by category first to make selection cleaner
    categories = ["All Categories"] + sorted(recommender.df["category"].unique().tolist())
    selected_category = st.sidebar.selectbox("Filter Products by Category", categories)
    
    if selected_category == "All Categories":
        filtered_df = recommender.df
    else:
        filtered_df = recommender.df[recommender.df["category"] == selected_category]

    # Create a nice format for product items in the dropdown
    product_options = filtered_df.apply(lambda r: f"{r['name']} ({r['id']})", axis=1).tolist()
    
    if not product_options:
        st.sidebar.warning("No products found in this category.")
        return

    selected_product_str = st.sidebar.selectbox("Select Target Product", product_options)
    
    # Extract the ID from the selected product string (format: "Name (ID)")
    selected_product_id = selected_product_str.split(" (")[-1][:-1]
    
    # 2. Recommendation Count Slider
    num_recommendations = st.sidebar.slider("Number of Recommendations", min_value=3, max_value=10, value=5)

    # Fetch details of target product
    target_product = recommender.df[recommender.df["id"] == selected_product_id].iloc[0]

    # Layout split: Left side for selected product, Right side for Recommendations
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎯 Selected Product")
        st.markdown(f"""
            <div class="product-card">
                <span class="product-category">{target_product['category']}</span>
                <div class="product-name">{target_product['name']}</div>
                <div style="font-size: 0.85rem; color: #6c757d; margin-bottom: 0.75rem;">ID: {target_product['id']}</div>
                <div class="product-description">{target_product['description']}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Display the explanation of model variables for capstone
        with st.expander("ℹ️ Capstone Project Info"):
            st.markdown("""
                ### System Architecture
                This system operates as a **Content-Based Filtering** recommender:
                1. **TF-IDF Vectorizer**: Translates raw text descriptions into numerical vectors by assigning weights to words based on their frequency (TF) and inverse document frequency (IDF).
                2. **Cosine Similarity**: Measures the cosine angle between two TF-IDF vectors in high-dimensional space:
                $$\\text{Similarity}(A, B) = \\cos(\\theta) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$$
                3. **Explainable AI (XAI)**: Displays the top terms matching both vectors to make the recommendation logic transparent to the user.
            """)

    with col2:
        st.subheader(f"🚀 Top {num_recommendations} Recommendations")
        
        # Get recommendations
        with st.spinner("Calculating similarity scores..."):
            recommendations_df = recommender.get_recommendations(selected_product_id, top_n=num_recommendations)
        
        # Display each recommendation
        for idx, row in recommendations_df.iterrows():
            score_percentage = row['similarity_score']
            progress_val = float(score_percentage)
            
            # Format display
            st.markdown(f"""
                <div class="rec-card">
                    <span class="rec-score-badge">Match: {format_score(score_percentage)}</span>
                    <span class="product-category">{row['category']}</span>
                    <div class="product-name">{row['name']}</div>
                    <div style="font-size: 0.85rem; color: #6c757d; margin-bottom: 0.75rem;">ID: {row['id']}</div>
                    <div class="product-description">{row['description']}</div>
                    <div class="reason-box">
                        💡 <strong>Recommendation Reason:</strong> {row['explanation']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 3. Interactive debug/presentation section "Under the Hood"
    st.subheader("📊 Under the Hood: NLP Feature Analysis")
    st.markdown("Explore how the TF-IDF representation extracts features from the selected product's text description.")
    
    tab1, tab2 = st.tabs(["Selected Product TF-IDF Features", "Recommendation Score Distribution"])
    
    with tab1:
        # Get TF-IDF vector for the selected product and display top words
        target_idx = recommender.df.index[recommender.df["id"] == selected_product_id].tolist()[0]
        v_target = recommender.tfidf_matrix[target_idx].toarray()[0]
        feature_names = recommender.vectorizer.get_feature_names_out()
        
        # Create a dataframe of feature weights
        features_df = pd.DataFrame({
            "Term": feature_names,
            "TF-IDF Weight": v_target
        })
        # Filter out zero weights and sort
        features_df = features_df[features_df["TF-IDF Weight"] > 0].sort_values(by="TF-IDF Weight", ascending=False).reset_index(drop=True)
        
        if not features_df.empty:
            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                # Plot the top features using st.bar_chart
                chart_data = features_df.head(10).set_index("Term")
                st.bar_chart(chart_data)
            with col_table:
                st.write("Top TF-IDF Terms in Selected Product:")
                st.dataframe(features_df.head(10), use_container_width=True)
        else:
            st.write("No TF-IDF features found.")
            
    with tab2:
        # Show a summary/comparison chart of similarity scores for the recommendations
        chart_df = recommendations_df[["name", "similarity_score"]].copy()
        chart_df["similarity_score"] = chart_df["similarity_score"] * 100
        chart_df = chart_df.rename(columns={"name": "Product", "similarity_score": "Similarity %"}).set_index("Product")
        
        st.write("Visual Comparison of Similarity Scores across Recommendations:")
        st.bar_chart(chart_df)

if __name__ == "__main__":
    main()
