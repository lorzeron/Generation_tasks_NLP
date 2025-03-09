import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("model", exist_ok=True)
os.makedirs("plots", exist_ok=True)

def main():
    # Загружаем очищенный датасет
    df = pd.read_csv("data/dialogues_clean.csv", encoding="cp1251")
    # Удаляем строки с пустыми значениями в колонке 'dialogue_clean'
    df = df.dropna(subset=["dialogue_clean"])

    dialogues = df["dialogue_clean"].tolist()
    
    # Обучение TF‑IDF векторизатора
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(dialogues)
    
    # Сохранение модели и матрицы
    joblib.dump(vectorizer, "model/vectorizer.pkl")
    joblib.dump(tfidf_matrix, "model/tfidf_matrix.pkl")
    
    print("TF‑IDF модель обучена и сохранена в папке model/")
    
    # Построение графика распределения количества слов в репликах
    lengths = np.array([len(doc.split()) for doc in dialogues])
    plt.hist(lengths, bins=20, color='skyblue', edgecolor='black')
    plt.title("Распределение количества слов в репликах")
    plt.xlabel("Количество слов")
    plt.ylabel("Частота")
    plt.savefig("plots/dialogue_length_distribution.png")
    plt.close()
    print("График распределения длины реплик сохранён в plots/dialogue_length_distribution.png")

if __name__ == "__main__":
    main()
