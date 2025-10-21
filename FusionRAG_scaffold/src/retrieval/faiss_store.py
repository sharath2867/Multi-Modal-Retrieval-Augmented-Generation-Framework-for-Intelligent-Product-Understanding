
import argparse, os, pandas as pd, numpy as np
from sentence_transformers import SentenceTransformer
import faiss

def build_index(data_csv, index_path):
    df = pd.read_csv(data_csv)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = (df['title'].fillna('') + ' ' + df['bullet_points'].fillna('') + ' ' + df['description'].fillna('')).tolist()
    embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32))
    faiss.write_index(index, index_path)
    print(f"Built FAISS index with {len(texts)} items → {index_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", required=True)
    ap.add_argument("--index_path", required=True)
    ap.add_argument("--images_dir", default="")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
    build_index(args.data_csv, args.index_path)
