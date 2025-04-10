# agents/recommendation_agent.py

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Load the embedding model globally
model = SentenceTransformer('all-MiniLM-L6-v2')

def build_ticket_index(csv_path):
    """
    Loads a historical ticket CSV, processes the 'Issue Category' column,
    and builds a FAISS index for similarity search.
    """
    # Step 1: Load the CSV and skip bad lines
    df = pd.read_csv(csv_path, on_bad_lines='skip')

    # Step 2: Clean up column names
    df.columns = df.columns.str.strip()

    # Step 3: Use 'Issue Category' as the text to embed
    if 'Issue Category' not in df.columns:
        raise KeyError("Expected 'Issue Category' column not found. Available columns: " + str(df.columns.tolist()))

    ticket_texts = df['Issue Category'].fillna('').tolist()

    # Step 4: Convert text to embeddings
    embeddings = model.encode(ticket_texts, convert_to_tensor=False)

    # Step 5: Build FAISS index
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, ticket_texts, df


def recommend_solution(new_issue_summary, index, ticket_texts, df, top_k=1):
    """
    Takes a new issue summary, searches the index, and recommends the most relevant solution.
    """
    # Step 6: Encode the new issue summary
    query_vec = model.encode([new_issue_summary])[0]
    query_vec = np.array([query_vec])  # FAISS requires a 2D array

    # Step 7: Search the index
    D, I = index.search(query_vec, top_k)
    result_index = I[0][0]

    # Step 8: Retrieve and return the result
    matched_summary = ticket_texts[result_index]

    if 'Solution' not in df.columns:
        raise KeyError("Expected 'Solution' column not found. Available columns: " + str(df.columns.tolist()))

    recommended_resolution = df.iloc[result_index]['Solution']

    return matched_summary, recommended_resolution
