import numpy as np
import pandas as pd
from pathlib import Path
from umap import UMAP
from langchain_huggingface import HuggingFaceEmbeddings


def read_texts_with_color(file_path: str, color: str, min_length: int = 10) -> tuple[list[str], list[str]]:
    """Read texts from a file and assign a color"""
    with open(file_path, 'r') as f:
        texts = [line.strip() for line in f if len(line.strip()) >= min_length]
        colors = [color] * len(texts)

    if not texts:
        raise ValueError(f"No valid texts found in {file_path}")
    return texts, colors


def generate_embeddings(texts: list[str]) -> np.ndarray:
    """Generate embeddings using Hugging Face model"""
    embeddings_model = HuggingFaceEmbeddings()
    return np.array([embeddings_model.embed_query(text) for text in texts])


def reduce_dimensions(embeddings: np.ndarray) -> np.ndarray:
    """Reduce embeddings to 2D with UMAP"""
    return UMAP(
        n_components=2,
        n_neighbors=min(5, len(embeddings) - 1),
        metric='cosine',
        random_state=42
    ).fit_transform(embeddings)


def save_visualization_csv(texts: list[str], embeddings_2d: np.ndarray, colors: list[str], output_file: str):
    """Save results with color coding"""
    pd.DataFrame({
        'id': range(len(texts)),
        'text': texts,
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'color': colors
    }).to_csv(output_file, index=False)


def main(file_color_pairs: list[tuple[str, str]], output_file: str = "embeddings.csv"):
    """Main processing pipeline for multiple files"""
    try:
        all_texts = []
        all_colors = []

        # Read all files with their assigned colors
        for file_path, color in file_color_pairs:
            texts, colors = read_texts_with_color(file_path, color)
            all_texts.extend(texts)
            all_colors.extend(colors)

        print(f"Loaded {len(all_texts)} texts from {len(file_color_pairs)} files")

        # Generate and process embeddings
        embeddings = generate_embeddings(all_texts)
        embeddings_2d = reduce_dimensions(embeddings)

        # Save results
        save_visualization_csv(all_texts, embeddings_2d, all_colors, output_file)
        print(f"Saved visualization to {output_file}")
        print(f"Upload to Cosmograph: https://cosmograph.app/")

    except Exception as e:
        print(f"Error: {str(e)}")
        Path(output_file).unlink(missing_ok=True)


if __name__ == "__main__":
    input_files = [
        ("texts_file1.txt", "#dea9f5"),
        ("texts_file2.txt", "#68f7dd")
    ]

    main(file_color_pairs=input_files, output_file="embeddings.csv")