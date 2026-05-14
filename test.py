#!/usr/bin/env python3
"""
detect_duplicates.py

An optimized semantic duplicate detection engine using Ollama.

Key Architecture Improvements:
1. API Batching: Sends texts to Ollama in chunks rather than one-by-one.
2. Vectorized Math: Uses NumPy matrix multiplication instead of nested Python loops for instant O(1) matrix calculations.
3. Thresholding: Rather than just dumping a matrix, it isolates actual duplicate pairs based on a defined similarity threshold.
4. Nomic Prompting: Prepends "search_document:" (a requirement for optimal Nomic model performance).
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "nomic-embed-text:latest"
INPUT_FILE = Path("test_samples.json")
OUTPUT_CSV = Path("duplicate_report.csv")
CACHE_FILE = Path("embeddings_cache.json")

# The similarity score (0.0 to 1.0) above which records are flagged as duplicates.
# Nomic tends to group things tightly; 0.85 - 0.90 is a good starting point.
SIMILARITY_THRESHOLD = 0.8

# Nomic-embed-text specifically performs better when told what the text is for.
NOMIC_PREFIX = "search_document: "

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #
def load_samples(path: Path) -> List[Tuple[str, str]]:
    if not path.exists():
        print(f"Input file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [(item.get("title", ""), item.get("description", "")) for item in data]

def load_cache() -> Dict[str, List[float]]:
    if CACHE_FILE.exists():
        with CACHE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache: Dict[str, List[float]]) -> None:
    with CACHE_FILE.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Send a batch of strings to Ollama and return their embedding vectors."""
    if not texts:
        return []
        
    payload = {"model": MODEL_NAME, "input": texts}
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Error contacting Ollama: {e}", file=sys.stderr)
        sys.exit(1)

    # Note the plural "embeddings" here, which fixes the bug in the original script
    return resp.json().get("embeddings", [])

# --------------------------------------------------------------------------- #
# Main workflow
# --------------------------------------------------------------------------- #
def main() -> None:
    samples = load_samples(INPUT_FILE)
    
    # Combine title and description, adding the Nomic prefix
    texts = [f"{NOMIC_PREFIX}{title}\n{desc}".strip() for title, desc in samples]

    cache = load_cache()
    
    # 1. Identify what needs to be embedded vs what is cached
    texts_to_embed = [t for t in texts if t not in cache]
    
    # 2. Batch embed missing texts (Ollama's /api/embed natively handles array inputs)
    if texts_to_embed:
        print(f"Generating embeddings for {len(texts_to_embed)} new records...")
        new_embeddings = get_embeddings_batch(texts_to_embed)
        
        for text, emb in zip(texts_to_embed, new_embeddings):
            cache[text] = emb
        save_cache(cache)
    else:
        print("All records found in cache.")

    # 3. Assemble the ordered embedding matrix
    embeddings_list = [cache[t] for t in texts]
    # Convert to a 2D NumPy array for fast matrix math
    matrix_2d = np.array(embeddings_list) 

    # 4. Fast Cosine Similarity Matrix Computation
    print("Computing vectorized pairwise similarities...")
    
    # Normalize the vectors
    norms = np.linalg.norm(matrix_2d, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10 # Prevent division by zero
    normalized_matrix = matrix_2d / norms
    
    # Dot product of the matrix with its transpose yields the full similarity matrix instantly
    similarity_matrix = np.dot(normalized_matrix, normalized_matrix.T)

    # 5. Extract and format duplicates
    print(f"Flagging pairs with >{SIMILARITY_THRESHOLD * 100}% similarity...")
    
    duplicates_found = 0
    n = len(texts)
    
    with OUTPUT_CSV.open("w", encoding="utf-8") as f:
        f.write("Index_A,Index_B,Similarity_Score,Title_A,Title_B\n")
        
        # We only check the upper triangle of the matrix to avoid comparing A->B and B->A
        for i in range(n):
            for j in range(i + 1, n):
                score = float(similarity_matrix[i, j])
                
                if score >= SIMILARITY_THRESHOLD:
                    duplicates_found += 1
                    # Clean up titles to avoid breaking the CSV if they have commas
                    title_a = samples[i][0].replace(",", " ")
                    title_b = samples[j][0].replace(",", " ")
                    
                    f.write(f"{i},{j},{score:.4f},{title_a},{title_b}\n")
                    print(f"DUPLICATE DETECTED ({score:.2f}): [{title_a}] <--> [{title_b}]")

    print(f"\nDone. Found {duplicates_found} potential duplicate pairs.")
    print(f"Report saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()