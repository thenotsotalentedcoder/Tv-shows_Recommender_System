import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from fuzzywuzzy import process
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import streamlit as st
from PIL import Image
import requests
import os

# Set custom path for NLTK data
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')
nltk.data.path.append(nltk_data_path)

# Download NLTK data (only if not already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path, quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)

# Initialize stemmer and stop words
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

# Safe evaluation for string to list conversion
def safe_eval(x):
    try:
        return ast.literal_eval(x) if pd.notna(x) else []
    except (ValueError, SyntaxError):
        return []

# Extract 'name' from dictionary lists (for genres, etc.)
def extract_names(x):
    return ' '.join([i['name'] for i in x]) if isinstance(x, list) else ''

# Advanced preprocessing function
def advanced_preprocess(text):
    tokens = word_tokenize(str(text).lower())
    return ' '.join([stemmer.stem(token) for token in tokens if token.isalnum() and token not in stop_words])

# Normalize titles for matching
def normalize_title(title):
    return str(title).strip().lower() if pd.notna(title) else ""

# Data preprocessing function
def preprocess_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)

    # Fill missing values
    df['overview'] = df['overview'].fillna('')
    df['genres'] = df['genres'].fillna('[]')
    df['vote_average'] = df['vote_average'].fillna(0)
    df['vote_count'] = df['vote_count'].fillna(0)
    df['popularity'] = df['popularity'].fillna(0)
    df['number_of_seasons'] = df['number_of_seasons'].fillna(0)
    df['original_language'] = df['original_language'].fillna('')
    df['tagline'] = df['tagline'].fillna('')
    df['first_air_date'] = pd.to_datetime(df['first_air_date'], errors='coerce')

    # Convert 'genres' to list and extract names
    df['genres'] = df['genres'].apply(safe_eval).apply(extract_names)

    # Preprocess text data
    for col in ['overview', 'genres', 'tagline']:
        df[col] = df[col].apply(advanced_preprocess)

    # Normalize titles
    df['name'] = df['name'].fillna('').apply(normalize_title)

    # Combine relevant features for content similarity (adjusted weights)
    df['content_features'] = (
        df['name'] * 3 + ' ' +
        df['genres'] * 4 + ' ' +  # Increased genre weight
        df['overview'] * 3 + ' ' +  # Increased overview weight
        df['tagline']
    )

    # Calculate recency score
    df['recency_score'] = (df['first_air_date'].dt.year - df['first_air_date'].dt.year.min()) / (
        df['first_air_date'].dt.year.max() - df['first_air_date'].dt.year.min())

    # Remove rows with low votes and important missing data
    df = df[df['vote_count'] > 10]
    df = df.dropna(subset=['name', 'content_features'])

    df = df.reset_index(drop=True)
    return df

# Create sentence embeddings with consistent dimensions
def create_sentence_embeddings(df):
    model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(df['content_features'].tolist(), show_progress_bar=True)
    print(f"Sentence embeddings shape: {embeddings.shape}")

    # Debug: Ensure embeddings are of correct size (768)
    if embeddings.shape[1] != 768:
        raise ValueError(f"Expected embeddings dimension of 768, but got {embeddings.shape[1]}")

    return embeddings

# Create TF-IDF matrix
def create_tfidf_matrix(df):
    tfidf = TfidfVectorizer(preprocessor=advanced_preprocess)
    tfidf_matrix = tfidf.fit_transform(df['content_features'])
    return tfidf_matrix

# Create a show-to-index dictionary for lookups
def create_show_to_index_mapping(df):
    return pd.Series(df.index, index=df['name']).drop_duplicates()

# Get the closest match using fuzzy matching
def get_closest_title(user_input, show_titles):
    user_input = normalize_title(user_input)
    best_match, confidence = process.extractOne(user_input, show_titles)
    return best_match, confidence

# Function to calculate weighted similarity score
def calculate_weighted_similarity(sim_df, input_show):
    input_genres = set(input_show['genres'].split())
    input_language = input_show['original_language']
    input_type = 'tv_series' if input_show['number_of_seasons'] > 0 else 'movie'

    sim_df['genre_sim'] = sim_df['genres'].apply(
        lambda x: len(set(x.split()) & input_genres) / len(input_genres) if x else 0)
    sim_df['lang_match'] = (sim_df['original_language'] == input_language).astype(int)
    sim_df['type_match'] = ((sim_df['number_of_seasons'] > 0) == (input_type == 'tv_series')).astype(int)

    # Calculate weighted similarity
    sim_df['similarity'] = (
        0.25 * sim_df['similarity'] +  # Content similarity
        0.3 * sim_df['genre_sim'] +     # Genre similarity
        0.15 * sim_df['lang_match'] +   # Language match
        0.1 * sim_df['type_match'] +    # Type match
        0.1 * (sim_df['vote_average'] / 10) +  # Vote average
        0.05 * (sim_df['vote_count'] / sim_df['vote_count'].max()) + # Vote count
        0.05 * (np.log1p(sim_df['popularity']) / np.log1p(sim_df['popularity'].max()))  # Popularity
    )

    return sim_df

@st.cache_data  # Cache the data loading
def load_precomputed_data(data_file):
    try:
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        print(f"Loaded precomputed data from {data_file}")  # This will only print once
        return data
    except FileNotFoundError:
        print(f"Precomputed data file {data_file} not found.")
        return None

def get_recommendations(df, embeddings, tfidf_matrix, show_title, show_to_index, top_n=10):
    show_title = normalize_title(show_title)

    best_match, confidence = get_closest_title(show_title, show_to_index.index)
    print(f"Best match for '{show_title}' is '{best_match}' with confidence {confidence}%")

    if confidence < 60:
        return f"Show '{show_title}' not found or match confidence too low. Best match was '{best_match}'. Please try again."

    show_index = show_to_index[best_match]

    # Handle multiple entries for the same show title
    if isinstance(show_index, pd.Series):
        show_index = show_index.iloc[0]

    # Debugging output 
    print(f"show_index (after handling duplicates): {show_index}")
    print(f"Shape of embeddings[show_index] before reshape: {embeddings[show_index].shape}")

    if embeddings.shape[1] != 768:
        raise ValueError(f"Expected embeddings dimension of 768, but got {embeddings.shape[1]}")

    embedding_sim = cosine_similarity(embeddings[show_index].reshape(1, -1), embeddings)
    tfidf_sim = cosine_similarity(tfidf_matrix[show_index], tfidf_matrix)

    combined_sim = 0.6 * embedding_sim + 0.4 * tfidf_sim

    sim_scores = list(enumerate(combined_sim[0]))
    sim_scores = [(i, float(score)) for i, score in sim_scores]
    # sim_scores.insert(0, (show_index, 1.0))  # Remove this line 
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[:top_n + 1]  # Keep top_n + 1 to account for potential removal of input show

    show_indices_sim = [i[0] for i in sim_scores]

    input_show = df.iloc[show_index]

    sim_df = pd.DataFrame({
        'index': show_indices_sim,
        'similarity': [x[1] for x in sim_scores],
        'genres': df.iloc[show_indices_sim]['genres'],
        'original_language': df.iloc[show_indices_sim]['original_language'],
        'number_of_seasons': df.iloc[show_indices_sim]['number_of_seasons'],
        'vote_average': df.iloc[show_indices_sim]['vote_average'],
        'vote_count': df.iloc[show_indices_sim]['vote_count'],
        'popularity': df.iloc[show_indices_sim]['popularity'],
        'backdrop_path': df.iloc[show_indices_sim]['backdrop_path']  # Include backdrop_path
    })

    sim_df = calculate_weighted_similarity(sim_df, input_show)
    sim_df = sim_df.sort_values(by='similarity', ascending=False)

    # --- Modified section to handle duplicate titles and remove input show ---
    sim_df = sim_df[sim_df['index'] != show_index]  # Remove the input show from sim_df
    sim_df = sim_df.drop_duplicates(subset=['index'])
    recommendations = df.iloc[sim_df['index'].unique()][['name', 'backdrop_path']]

    # Merge similarity scores back into recommendations (using a left join)
    recommendations = pd.merge(recommendations, sim_df[['index', 'similarity']], left_index=True, right_on='index', how='left')
    recommendations = recommendations.set_index('index')  # Set 'index' as the index again
    # ---------------------------------------------------------------------

    recommendations = recommendations.head(top_n)  # Get the top_n recommendations

    return recommendations


# Function to save precomputed data
def save_precomputed_data(data, data_file):
    with open(data_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"Precomputed data saved to {data_file}")

def main():
    st.title("TV Shows Recommender")

    # Load precomputed data 
    dataset_file = 'TMDB_tv_dataset_v3.csv'
    embeddings_file = 'tv_embeddings.pkl'
    tfidf_file = 'tfidf_matrix.pkl'
    show_to_index_file = 'show_to_index.pkl'

    embeddings = load_precomputed_data(embeddings_file)
    tfidf_matrix = load_precomputed_data(tfidf_file)
    show_to_index = load_precomputed_data(show_to_index_file)

    if embeddings is None or tfidf_matrix is None or show_to_index is None:
        st.write("Precomputing data...")
        df = preprocess_data(dataset_file)
        embeddings = create_sentence_embeddings(df)
        save_precomputed_data(embeddings, embeddings_file)
        tfidf_matrix = create_tfidf_matrix(df)
        save_precomputed_data(tfidf_matrix, tfidf_file)
        show_to_index = create_show_to_index_mapping(df)
        save_precomputed_data(show_to_index, show_to_index_file)
    else:
        df = preprocess_data(dataset_file)

    if "user_input" not in st.session_state:
        st.session_state.user_input = ""  

    # --- Dropdown menu for show selection ---
    all_shows = sorted(show_to_index.index.tolist())
    all_shows.insert(0, "")  # Add empty string at the beginning
    selected_show = st.selectbox("Select a TV Show:", all_shows, key="show_select")

    # --- Check if a show has been selected (exclude empty string) ---
    if selected_show != "": 
        normalized_input = normalize_title(selected_show)
        if normalized_input not in show_to_index:
            st.write("Show not found in the dataset. Please select from the dropdown or enter another title.")
        else: 
            recommendations = get_recommendations(df, embeddings, tfidf_matrix, selected_show, show_to_index)

            if isinstance(recommendations, str):
                st.write(recommendations)
            else:
                st.subheader(f"Recommendations for '{selected_show}':")

                # Create columns for better layout
                cols = st.columns(5)

                # Iterate through recommendations and display in columns
                for i, (index, row) in enumerate(recommendations.iterrows()):
                    with cols[i % 3]:
                        show_name = row['name']
                        backdrop_path = row['backdrop_path']

                        if backdrop_path:
                            image_url = f"https://image.tmdb.org/t/p/w300{backdrop_path}"
                            try:
                                image = Image.open(requests.get(image_url, stream=True).raw)
                                st.image(image, caption=show_name, width=200)
                            except:
                                st.write(f"Could not load image for {show_name}")
                        else:
                            st.write(show_name)

if __name__ == '__main__':
    main()