import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch

def load_model_components(df_path, collection_name="medicines"):
    """
    Loads the DataFrame and connects to the existing ChromaDB.
    """
    # --- GPU CHECK ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Initializing Recommender (ChromaDB + MMR) on {device.upper()} ---")
    
    if not os.path.exists(df_path):
        print(f"\nFATAL ERROR: {df_path} not found. Run train_model.py first.")
        return None, None, None

    if not os.path.exists("chroma_db"):
         print("\nFATAL ERROR: 'chroma_db' folder not found. Run train_model.py first.")
         return None, None, None

    try:
        df = pd.read_pickle(df_path)
        
        # --- GPU MODIFICATION: Load model on GPU ---
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        chroma_client = chromadb.PersistentClient(path="chroma_db")
        collection = chroma_client.get_collection(name=collection_name)
        
        print("Recommender initialized successfully.")
        return model, collection, df
        
    except Exception as e:
        print(f"\nERROR loading components: {e}")
        return None, None, None

def mmr_sort(query_embedding, candidate_embeddings, candidate_ids, k=10, lambda_param=0.5):
    """
    Maximal Marginal Relevance (MMR) algorithm.
    """
    if not candidate_ids:
        return []

    query_embedding = np.array(query_embedding).reshape(1, -1)
    candidate_embeddings = np.array(candidate_embeddings)
    
    sim_to_query = cosine_similarity(query_embedding, candidate_embeddings)[0]
    
    selected_indices = []
    candidate_indices = list(range(len(candidate_ids)))
    
    while len(selected_indices) < k and candidate_indices:
        best_score = -float('inf')
        best_idx = -1
        
        for idx in candidate_indices:
            relevance = sim_to_query[idx]
            
            if not selected_indices:
                redundancy = 0
            else:
                selected_embeddings = candidate_embeddings[selected_indices]
                current_embedding = candidate_embeddings[idx].reshape(1, -1)
                redundancy = np.max(cosine_similarity(current_embedding, selected_embeddings))
            
            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            
            if score > best_score:
                best_score = score
                best_idx = idx
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            candidate_indices.remove(best_idx)
            
    return [candidate_ids[i] for i in selected_indices]

def get_recommendations(query, df, model, collection):
    """
    Gets medicine recommendations using Sentence Transformer embeddings 
    stored in ChromaDB, ranked via MMR.
    """
    try:
        # 1. Embed the user query
        query_vec = model.encode([query]).tolist()
        
        # 2. Query ChromaDB
        results = collection.query(
            query_embeddings=query_vec,
            n_results=20,
            include=['embeddings', 'documents', 'metadatas'] 
        )
        
        # Safety check: Ensure results are not None or empty
        if results is None:
            return pd.DataFrame()

        if not results.get('ids') or not results['ids'][0]:
            return pd.DataFrame()

        candidate_ids = results['ids'][0]
        candidate_embeddings = results['embeddings'][0]
        
        # 3. Apply MMR
        final_ids = mmr_sort(query_vec[0], candidate_embeddings, candidate_ids, k=10, lambda_param=0.5)
        
        # Convert IDs back to integers for DataFrame lookup
        final_indices = [int(uid) for uid in final_ids]
        return df.loc[final_indices]

    except Exception as e:
        print(f"Error during recommendation: {e}")
        return pd.DataFrame()

def get_substitutes(medicine_name, df):
    """
    Gets substitutes for a given medicine.
    """
    substitutes = df[df['name'] == medicine_name][['substitute0', 'substitute1', 'substitute2', 'substitute3', 'substitute4']]
    return substitutes.values.flatten().tolist()

# --- NEW FUNCTIONALITY: Contextual Recommendations ---
def get_contextual_recommendations(query, df, model, collection):
    """
    Finds medicines for 'associated conditions' or 'causes' related to the query.
    Uses a broader search query and higher diversity (lower lambda) in MMR.
    """
    try:
        # Augment query to find deeper associations (e.g. "Fever" -> "Fever causes associated conditions")
        advanced_query = query + " associated conditions causes treatment"
        
        query_vec = model.encode([advanced_query]).tolist()
        
        # Fetch more results to explore the 'tail' of the distribution
        results = collection.query(
            query_embeddings=query_vec,
            n_results=50, 
            include=['embeddings', 'documents', 'metadatas'] 
        )
        
        if not results.get('ids') or not results['ids'][0]:
            return pd.DataFrame()

        candidate_ids = results['ids'][0]
        candidate_embeddings = results['embeddings'][0]
        
        # Use lower lambda_param (0.3) to prefer DIVERSITY over strict similarity
        final_ids = mmr_sort(query_vec[0], candidate_embeddings, candidate_ids, k=15, lambda_param=0.3)
        
        final_indices = [int(uid) for uid in final_ids]
        return df.loc[final_indices]

    except Exception as e:
        print(f"Error getting contextual recommendations: {e}")
        return pd.DataFrame()

if __name__ == '__main__':
    # Test block
    DATAFRAME_FILE = 'processed_data.pkl'
    model, collection, df = load_model_components(DATAFRAME_FILE)
    if df is not None:
        print("Testing Contextual Search for 'Fever'...")
        recs = get_contextual_recommendations("Fever", df, model, collection)
        print(recs[['name', 'reason']].head(5))