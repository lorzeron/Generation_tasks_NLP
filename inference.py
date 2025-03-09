import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка векторизатора и матрицы TF‑IDF
vectorizer = joblib.load("model/vectorizer.pkl")
tfidf_matrix = joblib.load("model/tfidf_matrix.pkl")

# Загружаем оригинальный датасет (season1.csv) для получения исходных реплик
df_original = pd.read_csv("data/season1.csv", encoding="cp1251")
dialogues = df_original["line"].tolist()

def get_reply(query: str) -> str:
    query_clean = query.lower()  
    query_vec = vectorizer.transform([query_clean])
    cos_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()
    best_idx = cos_sim.argmax()
    return dialogues[best_idx]

if __name__ == "__main__":
    user_input = input("Введите ваш вопрос: ")
    reply = get_reply(user_input)
    print("Ответ:", reply)
