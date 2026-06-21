import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ProductRecommender:
    """
    A Content-Based Product Recommendation Engine.
    Uses TF-IDF (Term Frequency-Inverse Document Frequency) to vectorize text descriptions
    and Cosine Similarity to find the most similar products.
    Includes a feature to explain recommendations based on overlapping high-impact terms.
    """
    
    def __init__(self, data_path="data/sample_products.csv"):
        self.data_path = data_path
        self.df = None
        self.vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.load_data()
        self.fit_model()

    def load_data(self):
        """Loads and prepares the dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            # Ensure required columns exist
            required_cols = ["id", "name", "category", "description"]
            for col in required_cols:
                if col not in self.df.columns:
                    raise ValueError(f"Missing required column: {col} in dataset.")
            
            # Preprocess: fill missing descriptions with empty strings
            self.df["description"] = self.df["description"].fillna("")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def fit_model(self):
        """
        Fits the TF-IDF Vectorizer on the product descriptions
        and computes the Cosine Similarity matrix.
        """
        # Initialize the TF-IDF Vectorizer
        # - stop_words='english' removes common filler words (e.g., 'the', 'is', 'and')
        # - ngram_range=(1, 2) allows capture of single words and two-word phrases (e.g., 'active noise')
        # - min_df=1 keeps all terms that appear in at least 1 document
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        
        # Fit vectorizer and transform product descriptions into a matrix of TF-IDF features
        # Shape: (num_products, num_features)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["description"])
        
        # Compute pairwise Cosine Similarity scores between all products
        # Cosine similarity is the dot product of normalized TF-IDF vectors.
        # Range: [0, 1] where 1 is identical and 0 is completely dissimilar.
        # Shape: (num_products, num_products)
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)

    def explain_recommendation(self, target_idx, rec_idx, top_n_terms=3):
        """
        Explains why a product was recommended by finding the words that contributed 
        most to the similarity between target_idx and rec_idx.
        
        Mathematically, Cosine Similarity = dot_product(V_target, V_rec).
        The terms that maximize V_target[i] * V_rec[i] are the primary drivers of similarity.
        """
        # Convert sparse row vectors to dense numpy arrays
        v_target = self.tfidf_matrix[target_idx].toarray()[0]
        v_rec = self.tfidf_matrix[rec_idx].toarray()[0]
        
        # Calculate element-wise product of TF-IDF scores
        elementwise_product = v_target * v_rec
        
        # Get feature (word) names from the vectorizer
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Find indices of the highest product values that are greater than zero
        sorted_indices = np.argsort(elementwise_product)[::-1]
        
        reasons = []
        for idx in sorted_indices:
            if elementwise_product[idx] > 0:
                reasons.append(feature_names[idx])
            if len(reasons) >= top_n_terms:
                break
                
        # Return a friendly explanation string
        if reasons:
            # Capitalize each word for neatness
            formatted_terms = [f"**{word.title()}**" for word in reasons]
            return f"Both products mention key terms: {', '.join(formatted_terms)}."
        else:
            return "Recommended based on general category similarities."

    def get_recommendations(self, product_id, top_n=5):
        """
        Gets the top_n recommendations for a given product_id.
        Returns a DataFrame of recommended products with similarity scores and explanations.
        """
        # Find the index of the selected product
        target_indices = self.df.index[self.df["id"] == product_id].tolist()
        if not target_indices:
            raise KeyError(f"Product ID '{product_id}' not found in the database.")
        
        target_idx = target_indices[0]
        
        # Fetch similarity scores for the target product
        sim_scores = list(enumerate(self.similarity_matrix[target_idx]))
        
        # Sort products based on similarity score in descending order
        # sim_scores is a list of tuples: (index, score)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Exclude the target product itself (which is always index 0 with score 1.0)
        recommendations = []
        for idx, score in sim_scores:
            if idx == target_idx:
                continue
            
            # Get explanation reasons
            reason = self.explain_recommendation(target_idx, idx)
            
            # Fetch product details
            prod_info = self.df.iloc[idx].to_dict()
            prod_info["similarity_score"] = float(score)
            prod_info["explanation"] = reason
            
            recommendations.append(prod_info)
            
            if len(recommendations) >= top_n:
                break
                
        return pd.DataFrame(recommendations)

# Quick local test execution
if __name__ == "__main__":
    try:
        recommender = ProductRecommender()
        print("Model initialized successfully!")
        
        # Test with the first product
        test_id = recommender.df.iloc[0]["id"]
        test_name = recommender.df.iloc[0]["name"]
        print(f"\nGetting recommendations for: {test_name} (ID: {test_id})")
        
        recs = recommender.get_recommendations(test_id)
        for i, row in recs.iterrows():
            print(f"{i+1}. {row['name']} [{row['category']}] (Score: {row['similarity_score']:.3f})")
            print(f"   Reason: {row['explanation']}")
    except Exception as e:
        print(f"Error during local test: {e}")
