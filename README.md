# talk-to-esenin
Телеграм-бот, который отвечает стихами Сергея Есенина и генерит открытку с получившимся текстом.

@talktoesenin_bot

Технологии:

- Python + python-telegram-bot — для работы с Telegram
- Word2Vec (gensim) — семантическое векторное представление слов
- Cosine Similarity (scikit-learn) — поиск близких по смыслу отрывков
- Pillow (PIL) — генерация изображений с текстом
- chardet — корректное чтение файла в любой кодировке
