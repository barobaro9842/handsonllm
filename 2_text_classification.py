############## 1. 분류특화모델

from datasets import load_dataset

# 데이터셋 로드
data = load_dataset("rotten_tomatoes")
data

data["train"][0,-1]


# 파이프라인으로 모델 로드
from openai.types.chat import chat_completion
from torch import _test_serialization_subcmul
from transformers import pipeline

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

pipe = pipeline(
  model=model_path,
  tokenizer=model_path,
  return_all_scores=True,
  device="cuda:0"
)

# 추론
import numpy as np
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "text")), total=len(data["test"])):
  negative_score = output[0]["score"]
  positive_score = output[2]["score"]
  assignment = np.argmax([negative_score, positive_score])
  y_pred.append(assignment)

# 평가
from sklearn.metrics import classification_report

def evaluate_performance(y_true, y_pred):
  performance = classification_report(
    y_true, y_pred,
    target_names=["Negative Review", "Positive Review"]
  )
  print(performance)

evaluate_performance(data["test"]["label"], y_pred)


############## 2. Sentence Transformer (SBERTA)

from sentence_transformers import SentenceTransformer

# 모델 로드
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

train_embeddings = model.encode(list(data["train"]["text"]), show_progress_bar=True)
test_embeddings = model.encode(list(data["test"]["text"]), show_progress_bar=True)

train_embeddings.shape

# 로지스틱회귀 모델 훈련
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=42)
clf.fit(train_embeddings, data["train"]["label"])

y_pred = clf.predict(test_embeddings)
evaluate_performance(data["test"]["label"], y_pred)

# 분류기 사용X (코사인 유사도)
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

df = pd.DataFrame(np.hstack([train_embeddings, np.array(data["train"]["label"]).reshape(-1,1)]))
averaged_target_embeddings = df.groupby(768).mean().values
averaged_target_embeddings

sim_matrix = cosine_similarity(test_embeddings, averaged_target_embeddings)
y_pred = np.argmax(sim_matrix, axis=1)

evaluate_performance(data["test"]["label"], y_pred)

################## 3. fine tuning
pipe = pipeline(
  "text2text-generation",
  model="google/flan-t5-small",
  device="cuda:0"
)

prompt = "Is the following sentence positive or negative? "
data = data.map(lambda example: {"t5": prompt + example['text']})
data

evaluate_performance(data["test"]["label"], y_pred)

import openai
import os

# 환경 변수에서 API Key 가져오기
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 생성하거나 환경 변수를 설정해주세요.")

client = openai.OpenAI(api_key=api_key)

def chatgpt_generation(prompt, document, model="gpt-4o-mini"):
  messages = [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": prompt.replace("[DOCUMENT]", document)
    }
  ]
  
  chat_completion = client.chat.completions.create(
    messages=messages,
    model=model,
    temperature=0
  )
  
  return chat_completion.choices[0].message.content

prompt = """Predict whether the following document is a positive or negative movie review:
  
[DOCUMENT]

If it is positive return 1 and if it is negative return 0. Do not give any other answers/
"""

document = "unpretentious, charming, quirky, original"
chatgpt_generation(prompt, document)

