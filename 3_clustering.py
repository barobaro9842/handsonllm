from datasets import load_dataset

dataset = load_dataset("maartengr/arxiv_nlp")["train"]
dataset[0]

abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("thenlper/gte-small")
embedding = embedding_model.encode(list(abstracts), show_progress_bar=True)

embedding.shape

from umap import UMAP
