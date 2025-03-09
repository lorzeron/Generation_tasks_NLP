Generation_tasks_NLP/
|
|-- .gradio/                 # Директория для файлов Gradio
|   |-- certificate.pem      # Сертификат безопасности
|
|-- data/                    # Данные для обучения и тестирования
|   |-- dialogues_clean.csv  # Очищенные диалоги
|   |-- season1.csv         # Данные первого сезона
|
|-- model/                   # Файлы модели
|   |-- tfidf_matrix.pkl     # Матрица TF-IDF
|   |-- vectorizer.pkl      # Сериализованный векторизатор
|
|-- plots/                   # Графики и визуализации
|   |-- dialogue_length_distribution.png  # График распределения длины диалогов
|
|-- benchmark.py             # Скрипт для оценки производительности
|-- inference.py             # Код для инференса модели
|-- preprocess.py            # Скрипт предобработки данных
|-- requirements.txt         # Зависимости проекта
|-- train_model.py           # Код для обучения модели
|-- ui.py                    # Код интерфейса