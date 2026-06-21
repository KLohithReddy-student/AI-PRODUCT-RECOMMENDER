# AI-Based Product Recommendation System

![AI Product Recommendation System Banner](assets/project_thumbnail.png)

An AI-powered content-based product recommendation system built with **Python**, **Streamlit**, and **Scikit-Learn**. This application uses Natural Language Processing (NLP) techniques to compute textual similarities between product descriptions and suggest the most relevant items to a user.

This project is structured, thoroughly documented, and designed to serve as a stellar **Kaggle Capstone Project** or portfolio piece.

🚀 **Live Interactive Demo:** [ai-recommender-lohith.streamlit.app](https://ai-recommender-lohith.streamlit.app/)

---

## 🌟 Features

1. **Category Filtering & Product Selection**: Easily browse the catalog by category or select any product to query recommendations.
2. **Textual Content-Based Matching**: Recommends the top 5 (or user-defined range of) most similar products based on semantic details.
3. **Explainable AI (XAI)**: Displays a clear explanation detailing the exact overlapping keywords (e.g., *Wireless*, *Bass*, *Sound*) that drove the recommendation.
4. **Under-the-Hood Debugging Panel**: Displays the mathematical TF-IDF feature weights of the selected product and lists similarity score breakdowns using interactive charts.
5. **Clean & Modern UI**: Responsive card layouts, progress bar similarity indicators, and clear styling built entirely on top of Streamlit.

---

## 📐 Theoretical Framework (For Capstone Documentation)

### 1. Preprocessing & Tokenization
Text descriptions are first cleaned by converting all words to lowercase and filtering out English stop words (common helper words like *the*, *a*, *is*, *in* that do not carry semantic content) using a custom Scikit-Learn pipeline.

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)
TF-IDF calculates the importance of a term $t$ within a specific product description $d$ relative to the entire dataset (corpus) $D$:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

*   **Term Frequency (TF)**: The number of times a term appears in a document.
*   **Inverse Document Frequency (IDF)**: Measures how common or rare a term is across all documents:
    $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

This ensures that rare, highly descriptive words (e.g., "Bluetooth") get higher weights than general words (e.g., "features").

### 3. Cosine Similarity
To determine how similar two products $A$ and $B$ are, we represent them as high-dimensional TF-IDF vectors and calculate the cosine of the angle between them:

$$\text{Similarity}(A, B) = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$

A score of $1.0$ (or $100\%$) indicates identical descriptions, while $0.0$ indicates no shared features.

### 4. Recommendation Explanations
This system features an explainability algorithm. To identify which words contributed most to a recommendation, it calculates the element-wise product of the TF-IDF vectors for the target product $A$ and the recommended product $B$:

$$\text{Overlap}_i = A_i \times B_i$$

The feature dimensions with the largest non-zero product values are translated back to words, providing a transparent reason (e.g., "Both products mention key terms: **Noise-canceling**, **Headphones**").

---

## 📂 Project Structure

```
ai-product-recommender/
├── data/
│   └── sample_products.csv      # Automatically generated product catalog (32 products)
├── app.py                      # Main Streamlit web application dashboard
├── recommender.py              # Recommendation Engine class (TF-IDF & Cosine Similarity logic)
├── generate_data.py            # Script to generate sample catalog
├── requirements.txt            # Python packages required to run the project
└── README.md                   # Project documentation (this file)
```

---

## 🛠️ Setup & Installation

### Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### Step 1: Clone or Copy the Repository
Navigate to the project directory:
```bash
cd ai-product-recommender
```

### Step 2: Install Dependencies
Install all required libraries using `pip`:
```bash
pip install -r requirements.txt
```

### Step 3: Generate the Dataset
Create the sample dataset of products:
```bash
python generate_data.py
```
This generates the CSV file containing 30+ products under `data/sample_products.csv`.

---

## 🚀 Running the Web Application

To launch the interactive dashboard, run:
```bash
streamlit run app.py
```

The application will automatically start in your default web browser (typically at `http://localhost:8501`).

---

## 💡 Extending the Capstone Project

If you are presenting this for a Kaggle project, here are some excellent ways to expand it further:
1.  **Collaborative Filtering Integration**: Add an alternative page that recommends products based on user ratings using Singular Value Decomposition (SVD) from the `surprise` library.
2.  **Hybrid Recommender**: Combine this content-based similarity score with user ratings to form a hybrid score.
3.  **Deep Learning Embeddings**: Replace TF-IDF vectors with dense word embeddings from pre-trained models like Word2Vec, GloVe, or BERT (via Hugging Face's `transformers` or `sentence-transformers`) for semantic similarity (e.g., understanding that "phone" and "mobile" are related).
4.  **Database Integration**: Connect the data loading mechanism in `recommender.py` to a database (like SQLite or PostgreSQL) rather than a CSV file.
