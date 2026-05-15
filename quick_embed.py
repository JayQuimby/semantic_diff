import json
import csv
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from pathlib import Path
from typing import List

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODEL_DIR = Path("./models")
ONNX_MODEL = MODEL_DIR / "model_quantized.onnx"
INPUT_FILE = Path("./data/test_samples.json")
OUTPUT_CSV = Path("./data/duplicate_report.csv")

SIMILARITY_THRESHOLD = 0.85
MAX_LENGTH = 512

# --------------------------------------------------------------------------- #
# Embedding & Search Engine
# --------------------------------------------------------------------------- #
class SemanticEngine:
    def __init__(self):
        tokenizer_path = MODEL_DIR / "tokenizer.json"
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=MAX_LENGTH)
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self.session = ort.InferenceSession(str(ONNX_MODEL))

    def encode(self, texts: List[str]) -> np.ndarray:
        encoded = self.tokenizer.encode_batch(texts)
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids)
        outputs = self.session.run(
            ["sentence_embedding"],
            {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids},
        )
        return outputs[0]

    @staticmethod
    def compute_similarity(matrix: np.ndarray) -> np.ndarray:
        """Vectorized cosine similarity for the entire matrix."""
        # FastEmbed vectors are usually pre-normalized, 
        # but we normalize here to be safe and 'standard'.
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized = matrix / norms
        return np.dot(normalized, normalized.T)

    @staticmethod
    def search(query_vector: np.ndarray, corpus_matrix: np.ndarray, top_k: int = 5):
        """Standard semantic search: find top_k matches for a query vector."""
        # Ensure query is 2D
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
            
        # Standardize normalization
        q_norm = query_vector / np.linalg.norm(query_vector)
        c_norm = corpus_matrix / np.linalg.norm(corpus_matrix, axis=1, keepdims=True)
        
        scores = np.dot(q_norm, c_norm.T).flatten()
        idx = np.argsort(scores)[::-1][:top_k]
        return idx, scores[idx]

# --------------------------------------------------------------------------- #
# Main Workflow
# --------------------------------------------------------------------------- #
def main():
    # 1. Load Data
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        samples = json.load(f)
    
    # BGE models don't strictly require a prefix for duplicate detection,
    # but for asymmetrical retrieval, you'd use "Represent this sentence for searching relevant passages: "
    texts = [f"{s.get('title', '')} {s.get('description', '')}".strip() for s in samples]
    titles = [s.get('title', 'No Title') for s in samples]

    # 2. Initialize Engine & Generate Embeddings
    print(f"Initializing Engine... (Model files expected in {MODEL_DIR})")
    engine = SemanticEngine()
    
    print(f"Embedding {len(texts)} records locally via FastEmbed...")
    embeddings = engine.encode(texts)
    print(f"Generated embeddings with shape: {embeddings.shape}")

    # 3. Compute Similarities
    print("Computing similarity matrix...")
    sim_matrix = engine.compute_similarity(embeddings)
    print(f"Similarity matrix shape: {sim_matrix.shape}")

    # 4. Extract Duplicates (Optimized via NumPy)
    # Use np.triu_indices to get only the upper triangle (avoiding self-match and mirrors)
    rows, cols = np.triu_indices(sim_matrix.shape[0], k=1)
    print(f"Total pairs to evaluate: {len(rows)}")

    # Filter indices based on threshold
    mask = sim_matrix[rows, cols] >= SIMILARITY_THRESHOLD
    dup_rows, dup_cols = rows[mask], cols[mask]
    scores = sim_matrix[dup_rows, dup_cols]

    # 5. Report Results
    print(f"Found {len(dup_rows)} potential duplicate pairs.")
    
    with OUTPUT_CSV.open("w", encoding="utf-8", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index_A", "Index_B", "Score", "Title_A", "Title_B"])
        
        for r, c, score in zip(dup_rows, dup_cols, scores):
            writer.writerow([r, c, f"{score:.4f}", titles[r], titles[c]])
            print(f"[{score:.2f}] Duplicate: '{titles[r]}' == '{titles[c]}'")

    print(f"Full report saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()