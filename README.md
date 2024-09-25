# TV Show Recommender System Documentation

## Overview

This document provides a comprehensive explanation of an advanced TV show recommender system implemented in Python. The system uses a hybrid approach combining content-based filtering and collaborative filtering techniques to suggest TV shows similar to a user's input. It's designed as a Streamlit web application for easy user interaction.

## Table of Contents

1. [Dependencies](#1-dependencies)
2. [Data Preprocessing](#2-data-preprocessing)
3. [Feature Engineering](#3-feature-engineering)
4. [Similarity Calculation](#4-similarity-calculation)
5. [Recommendation Generation](#5-recommendation-generation)
6. [Performance Optimization](#6-performance-optimization)
7. [User Interface](#7-user-interface)
8. [Main Execution Flow](#8-main-execution-flow)

## 1. Dependencies

The system relies on several Python libraries:

- pandas: For data manipulation and analysis
- numpy: For numerical computations
- scikit-learn: For machine learning utilities and TF-IDF vectorization
- nltk: For natural language processing tasks
- fuzzywuzzy: For string matching
- sentence-transformers: For creating sentence embeddings
- pickle: For serializing and deserializing Python objects
- streamlit: For creating the web application interface
- PIL (Python Imaging Library): For image processing
- requests: For making HTTP requests to fetch images

## 2. Data Preprocessing

### 2.1 Data Loading and Cleaning

The `preprocess_data` function handles the initial data preparation:

- Loads data from a CSV file
- Fills missing values in various columns
- Converts date strings to datetime objects
- Removes rows with low vote counts and missing crucial data

### 2.2 Text Preprocessing

The `advanced_preprocess` function performs text preprocessing:

- Tokenizes text
- Converts to lowercase
- Removes stopwords
- Applies stemming

This preprocessing is applied to 'overview', 'genres', and 'tagline' columns.

### 2.3 Genre Handling

The `safe_eval` and `extract_names` functions process the 'genres' column:

- Safely converts string representations of lists to actual lists
- Extracts genre names from dictionaries within these lists

## 3. Feature Engineering

### 3.1 Content Features

A combined 'content_features' column is created by concatenating:

- Show name (weighted 3x)
- Genres (weighted 4x)
- Overview (weighted 3x)
- Tagline

### 3.2 Recency Score

A 'recency_score' is calculated based on the show's first air date, normalizing it between 0 and 1.

## 4. Similarity Calculation

The system uses two methods to calculate similarity:

### 4.1 Sentence Embeddings

The `create_sentence_embeddings` function uses the SentenceTransformer model 'all-mpnet-base-v2' to create dense vector representations of the content features.

### 4.2 TF-IDF Vectorization

The `create_tfidf_matrix` function creates a TF-IDF (Term Frequency-Inverse Document Frequency) matrix from the content features.

## 5. Recommendation Generation

The `get_recommendations` function is the core of the recommendation process:

1. Uses fuzzy matching to find the closest match to the user's input
2. Calculates cosine similarity using both sentence embeddings and TF-IDF vectors
3. Combines these similarities with a 60-40 weight
4. Applies additional weighting factors:
   - Genre similarity (30%)
   - Language match (15%)
   - Show type match (TV series vs. movie) (10%)
   - Vote average (10%)
   - Vote count (5%)
   - Popularity (5%)
5. Sorts and returns the top N recommendations

## 6. Performance Optimization

### 6.1 Precomputation

The system uses precomputation to improve runtime performance:

- Preprocessed data, embeddings, TF-IDF matrix, and show-to-index mapping are saved to disk
- These are loaded on subsequent runs, avoiding repeated costly computations

### 6.2 Efficient Lookups

A show-to-index dictionary is created for quick title lookups.

### 6.3 Data Caching

Streamlit's `@st.cache_data` decorator is used to cache the loading of precomputed data, improving the app's performance across multiple user sessions.

## 7. User Interface

The system uses Streamlit to create a user-friendly web interface:

### 7.1 Show Selection

- A dropdown menu populated with all available TV shows
- Users can select a show from the dropdown or enter a title manually

### 7.2 Recommendation Display

- Recommendations are displayed in a grid layout
- Each recommendation includes the show's title and backdrop image (if available)
- Images are fetched from TMDB's API and displayed using PIL

## 8. Main Execution Flow

The `main` function orchestrates the system's operation:

1. Sets up the Streamlit interface
2. Attempts to load precomputed data
3. If not available, processes the data and saves the results
4. Displays the show selection dropdown
5. When a show is selected:
   - Generates recommendations
   - Displays recommendations with images in a grid layout

## Conclusion

This TV show recommender system employs a sophisticated hybrid approach, combining content-based features with collaborative filtering signals. By using advanced NLP techniques like sentence embeddings alongside traditional methods like TF-IDF, it aims to provide relevant and diverse recommendations. The system also considers factors beyond mere content similarity, such as show popularity and user ratings, to enhance the quality of recommendations. The Streamlit-based user interface makes it easy for users to interact with the system and discover new TV shows based on their preferences.
