# Recommendation System - Predefined Items

import streamlit as st
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# URL to the dataset
DATA_URL = "https://raw.githubusercontent.com/fenago/datasets/refs/heads/main/sample-data.csv"

@st.cache_data
def load_data():
    """
    Loads product data from the given URL and caches it for efficiency.
    """
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception as e:
        st.error(f"Error loading the data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("The dataset is empty. Try again later.")
    st.stop()


st.title("🛍 The Ultimate Product Recommendation System")

st.markdown(
    """
    *Looking for great recommendations on products you like?*  

    Pick a product, and we'll do our best to **WOW** you with incredible recommendations!  
    """
)


dark_mode = st.sidebar.checkbox("Enable Dark Mode") # Still need some work on the structure of this mode.

if dark_mode:
    st.markdown(
        """
        <style>
        
        body, .stApp {
            background-color: #121212 !important;
            color: #FFFFFF !important;
        }

        .css-1d391kg, .css-1lcbmhc, .css-1fcdlh3, .stSidebar {
            background-color: #121212 !important;
            color: white !important;
            border: none !important;
            box-shadow: black !important;
        }
        
        .stSelectbox, .stTextInput, .stTextArea, .stSlider, .stButton, 
        .stNumberInput, .stRadio, .stCheckbox, .stMultiSelect, .stDateInput {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
            border-radius: 8px;
            border: 1px solid #1E90FF !important;  
        }

        /* Dropdown labels, slider text, review input text */
        label, .st-eb, .st-c2, .st-df {
            color: white !important;
        }

        /* Buttons */
        .stButton>button {
            background-color: #1E90FF !important;  
            color: #000000 !important;
            border-radius: 8px;
            font-weight: bold;
        }

        /* Sidebar Buttons */
        .stButton>button:hover {
            background-color: #0D6EFD !important;  
        }

        /*Surprise Me! Button */
        .stButton>button {
            background-color: #1E90FF !important;  
            color: white !important;
        }

        /* Expanders */
        .st-expander {
            background-color: #252525 !important;
            color: white !important;
        }

        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #1E90FF !important;  
        }

        /* Links */
        a {
            color: #00BFFF !important;  
        }

        </style>
        """,
        unsafe_allow_html=True
    )



st.sidebar.header("Feeling Adventurous?")
if st.sidebar.button("Surprise Me!"):
    random_product = df.sample(1).iloc[0]
    
    st.sidebar.subheader(f"Random Pick: {random_product['id']}")

    # Extract product description
    random_description = random_product["description"]

    # Replace <br> tags with actual line breaks for better formatting
    random_description = random_description.replace("<br><br>", "\n\n")

    # Organizing bullet points properly
    random_description = random_description.replace("<ul>", "").replace("</ul>", "")
    random_description = random_description.replace("<li>", "- ").replace("</li>", "\n")

    # Display formatted description
    st.sidebar.markdown(f"Description:\n\n{random_description}", unsafe_allow_html=True)



product_id = st.selectbox("Select a Product:", df['id'].unique())

selected_product = df[df['id'] == product_id].iloc[0]

st.subheader(f"You Selected: {selected_product['id']}")

# Extract product description
description = selected_product["description"]

# Replace <br> tags with actual line breaks for better formatting
description = description.replace("<br><br>", "\n\n")

# Organizing bullet points properly
description = description.replace("<ul>", "").replace("</ul>", "")
description = description.replace("<li>", "- ").replace("</li>", "\n")

# Using markdown to display organized description
st.markdown(f"Description:\n\n{description}", unsafe_allow_html=True)



st.subheader("Rate & Review This Product")

# Star Rating
rating = st.slider("Rate this product (1 to 5 stars):", 1, 5, 3)

# User Review Input
review = st.text_area("Write a quick review about this product:")

if st.button("Submit Review"):
    st.success("Thank you for your feedback!")
    st.balloons()
    st.write(f"*You rated this product {rating} out of 5!*")
    if review:
        st.write(f"Your Review: {review}")
    else:
        st.write("You didn't write a review, but we appreciate your rating!")


@st.cache_data
def find_related_products(product_id, df, num_recommendations=3):
    """
    Uses machine learning (TF-IDF & Cosine Similarity) to find products with similar descriptions.
    """
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(df["description"].fillna(""))
        similarity_scores = cosine_similarity(tfidf_matrix)
        selected_idx = df[df['id'] == product_id].index[0]
        related_indices = similarity_scores[selected_idx].argsort()[-(num_recommendations + 1):-1][::-1]
        return df.iloc[related_indices]
    except Exception as e:
        st.error(f"Error finding related products: {e}")
        return pd.DataFrame()

related_products = find_related_products(product_id, df)


st.subheader("You Might Also Like:")
if not related_products.empty:
    for _, row in related_products.iterrows():
        st.subheader(f"Recommended: {row['id']}")

        # Extract and clean product description
        description = row["description"]
        description = description.replace("<br><br>", "\n\n")  # Fix line breaks
        description = description.replace("<ul>", "").replace("</ul>", "")
        description = description.replace("<li>", "- ").replace("</li>", "\n")

        # Use an expander for cleaner presentation
        with st.expander("View Product Description"):
            st.markdown(f"Description:\n\n{description}", unsafe_allow_html=True)

        st.markdown("---")  # Separator for better readability
else:
    st.write("No recommendations found. Try another product!")



st.sidebar.subheader("Search for a Product")
search_query = st.sidebar.text_input("Type a keyword and hit Enter:")

if search_query:
    search_results = df[df["description"].str.contains(search_query, case=False, na=False)]
    if not search_results.empty:
        st.sidebar.write(f"Found *{len(search_results)}* matching products:")
        for _, row in search_results.iterrows():
            st.sidebar.write(f"- {row['id']}: {row['description'][:50]}...")
    else:
        st.sidebar.write("No matching products found. Try another keyword!")



st.markdown(
    """
    ---
    *Warning:* This app may lead to an addiction to amazing product recommendations. Proceed with excitement! 

    *If you find something you love, don't forget to come back to discover more surprises.*
    """
)