import pandas as pd
import numpy as np
from difflib import get_close_matches


df=pd.read_csv('C:/Users/Administrator/OneDrive/Desktop/Data Projects/books.csv', on_bad_lines='skip')

# Fill missing values for soup components
df['title'] = df['title'].fillna('')
df['authors'] = df['authors'].fillna('')
df['publisher'] = df['publisher'].fillna('')

# Create soup column
df['soup'] = df['title'] + ' ' + df['authors'] + ' ' + df['publisher']

indices = pd.Series(df.index, index=df['title']).drop_duplicates()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['soup'])

from sklearn.metrics.pairwise import cosine_similarity

cosine_sim = cosine_similarity(tfidf_matrix)


def get_content_recommendations(title, cosine_sim=cosine_sim):
    idx = indices.get(title)

    if idx is None:
        return f"'{title}' not found in the dataset."

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    book_indices = [i[0] for i in sim_scores[1:11]]
    return df['title'].iloc[book_indices]

    from difflib import get_close_matches

def get_recommendations_with_scores(title, cosine_sim=cosine_sim, top_n=10):
    idx = indices.get(title)

    if idx is None:
        # Try to find a similar title using fuzzy matching
        possible_matches = get_close_matches(title, indices.keys(), n=1, cutoff=0.6)

        if possible_matches:
            # Retry with the closest match
            matched_title = possible_matches[0]
            idx = indices.get(matched_title)
            print(f"'{title}' not found. Showing results for '{matched_title}' instead.")
        else:
            # Return fallback recommendations (e.g. top-rated books)
            fallback = df.sort_values(by='rating', ascending=False).head(top_n)
            return [(row['title'], row['rating']) for _, row in fallback.iterrows()]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    book_indices_scores = sim_scores[1:top_n+1]

    results = [(df.iloc[i]['title'], score) for i, score in book_indices_scores]
    return results

def recommend_by_topic(keyword, top_n=5):
    # Search for the keyword in the book title
    topic_books = df[df['title'].str.contains(keyword, case=False, na=False)]

    if topic_books.empty:
        return f"No books found for topic '{keyword}'."

    # You can sort by average rating or ratings_count
    top_books = topic_books.sort_values(by='average_rating', ascending=False).head(top_n)

    return list(zip(top_books['title'], top_books['average_rating']))

import streamlit as st

st.title("📚 Book Recommendation System")

option = st.selectbox("Choose how you'd like to search:", ["By Book Title", "By Topic Keyword"])

if option == "By Book Title":
    title_input = st.text_input("Enter a book title:")
    if title_input:
        results = get_recommendations_with_scores(title_input, cosine_sim)
        st.subheader("Recommended Books:")
        for title, score in results:
            st.write(f"📖 {title} (Similarity Score: {round(score, 3)})")

elif option == "By Topic Keyword":
    topic_input = st.text_input("Enter a topic (e.g., IT, Business):")
    if topic_input:
        results = recommend_by_topic(topic_input)
        st.subheader("Top Books on This Topic:")
        if isinstance(results, str):
            st.write(results)
        else:
            for title, rating in results:
                st.write(f"📘 {title} (Rating: {rating})")

                print(df.columns)

