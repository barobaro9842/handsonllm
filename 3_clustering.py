from datasets import load_dataset

dataset = load_dataset("maartengr/arxiv_nlp")["train"]
dataset[0]

abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

# 임베딩 차원축소 (UMAP)
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("thenlper/gte-small")
embeddings = embedding_model.encode(list(abstracts), show_progress_bar=True)

embeddings.shape

from umap import UMAP

umap_model = UMAP(
  n_components=5,
  min_dist=0.0,
  metric='cosine',
  random_state=42
)

reduced_embeddings = umap_model.fit_transform(embeddings)
reduced_embeddings.shape

# 클러스터링(HDBSCAN)
from hdbscan import HDBSCAN

hdbscan_model = HDBSCAN(min_cluster_size=50).fit(reduced_embeddings)
clusters = hdbscan_model.labels_

len(set(clusters))

# 클러스터 조사
import numpy as np

cluster = 0
for index in np.where(clusters==cluster)[0][:3]:
  print(list(abstracts)[index][:300] + "... \n")

# 클러스터 시각화(2차원 차원 축소)
import pandas as pd

reduced_embeddings = UMAP(
  n_components=2,
  min_dist=0,
  metric="cosine",
  random_state=42
).fit_transform(embeddings)

df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
df["title"] = titles
df["cluster"] = [str(c) for c in clusters]

clusters_df = df.loc[df.cluster != "-1", :]
outliers_df = df.loc[df.cluster == "-1", :]

import matplotlib.pyplot as plt

plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2, c="grey")
plt.scatter(clusters_df.x, clusters_df.y, c=clusters_df.cluster.astype(int), alpha=0.05, cmap="tab20b")
plt.axis("off")
plt.show()

## BERTopic
from bertopic import BERTopic

topic_model = BERTopic(
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  verbose=True
).fit(abstracts, embeddings)

topic_model.get_topic_info()
topic_model.get_topic(0)
topic_model.find_topics("topic modeling")
topic_model.get_topic(25)
topic_model.topics_[titles.index("BERTopic: Neural topic modeling with a class-based TF-IDF procedure")]

fig = topic_model.visualize_documents(
  list(titles),
  reduced_embeddings=reduced_embeddings,
  width=1200,
  hide_annotations=True
)

fig.update_layout(font=dict(size=16))
fig.show()

fig = topic_model.visualize_barchart()
fig = topic_model.visualize_heatmap(n_clusters=50)
fig = topic_model.visualize_hierarchy()
fig.show()

from copy import deepcopy
original_topics = deepcopy(topic_model.topic_representations_)

def topic_differences(model, original_topics, nr_topics=5):
  df = pd.DataFrame(columns=["토픽","원본","업데이트"])
  for topic in range(nr_topics):
    
    og_words = " | ".join(list(zip(*original_topics[topic]))[0][:5])
    new_words = " | ".join(list(zip(*model.get_topic(topic)))[0][:5])
    df.loc[len(df)] = [topic, og_words, new_words]
  
  return df

# KeyBERTInspired
from bertopic.representation import KeyBERTInspired

representation_model = KeyBERTInspired()
topic_model.update_topics(abstracts, representation_model=representation_model)

topic_differences(topic_model, original_topics)

# MMR (토픽 다양화)
from bertopic.representation import MaximalMarginalRelevance

representation_model = MaximalMarginalRelevance(diversity=0.2)
topic_model.update_topics(abstracts, representation_model=representation_model)

topic_differences(topic_model, original_topics)