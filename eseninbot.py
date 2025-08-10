import re # для работы с регулярными выражениями (поиск слов, проверка знаков препинания)
import io # для работы с буферами в памяти (например, сохранение изображения без записи на диск)
import numpy as np # для численных операций, особенно с векторами
import random # для случайного выбора (например, фонового изображения)
from gensim.models import Word2Vec # модель для векторного представления слов (Word2Vec)
from sklearn.metrics.pairwise import cosine_similarity # функция для вычисления схожести векторов (по косинусу угла)
from PIL import Image, ImageDraw, ImageFont # для создания и редактирования изображений
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    filters,
) # библиотеки для взаимодействия с Telegram Bot API
import chardet # определяет кодировку файла (полезно, если неизвестно, в какой кодировке текст)

TOKEN = "<TOKEN>"

# === Загружаем корпус ===
with open("esenin.txt", "rb") as f:
    rawdata = f.read()

result = chardet.detect(rawdata)
encoding = result['encoding']
print(f"Detected encoding: {encoding}")

with open("esenin.txt", "r", encoding=encoding) as f:
    corpus = [line.rstrip("\n") for line in f]

# Токенизация для обучения Word2Vec
tokenized_corpus = [line.lower().split() for line in corpus if line.strip()]

'''
Подготовка текста для обучения модели Word2Vec:

- Каждая непустая строка переводится в нижний регистр и разбивается на слова (по пробелам).
- Получается список списков: каждая строка — это список слов.
- Пустые строки игнорируются
'''

w2v_model = Word2Vec(
    sentences=tokenized_corpus,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
)

'''
Обучение модели Word2Vec:

- sentences — подготовленный корпус (токенизированные строки).
- vector_size=100 — размер вектора для каждого слова (100 чисел).
- window=5 — контекст: модель учитывает до 5 слов слева и справа от текущего.
- min_count=1 — все слова включаются, даже если встречаются один раз.
- workers=4 — количество потоков для ускорения обучения.

Результат: модель w2v_model, которая умеет представлять слова как векторы и находить семантическую близость между ними.
'''

corpus_vectors = np.array(
    [
        np.mean([w2v_model.wv[w] for w in line.lower().split() if w in w2v_model.wv], axis=0)
        if any(w in w2v_model.wv for w in line.lower().split())
        else np.zeros(w2v_model.vector_size)
        for line in corpus
    ]
)

'''
Создание векторов для каждой строки корпуса:

Для каждой строки стихотворения:
- Берём все её слова, которые есть в модели Word2Vec.
- Получаем векторы этих слов.
- Усредняем их — получаем средний вектор строки (представление строки в векторном пространстве).
Если ни одно слово строки не найдено в модели, возвращается нулевой вектор.
Все векторы собираются в массив corpus_vectors — это матрица, где каждая строка — вектор строки из корпуса.
'''

def is_nonsense(text):
    clean = re.sub(r"[^а-яА-Яa-zA-Z\s]", "", text).strip()
    return len(clean.split()) < 2 or len(clean) < 3

'''
Проверка, является ли сообщение "бессмыслицей":

Удаляются все символы, кроме букв и пробелов.
Обрезаются пробелы по краям.
Если осталось меньше 2 слов или меньше 3 символов — текст считается бессмысленным.
Примеры: "привет", "как дела" — норма; "а", "123" — отклоняются.
'''

def is_ending(line):
    return bool(re.search(r"[.!?]$|\.{3}$", line.strip()))

'''
Проверка, заканчивается ли строка знаком окончания предложения:

Проверяет, заканчивается ли строка на ., !, ? или ....
Используется для определения, где заканчивается "отрывок" при извлечении.
'''

def extract_passage(start_idx):
    idx = start_idx
    while idx > 0 and corpus[idx - 1].strip() != "":
        idx -= 1
    passage = []
    for i in range(idx, len(corpus)):
        if corpus[i].strip() == "":
            continue
        passage.append(corpus[i])
        if is_ending(corpus[i]):
            break
    return passage

'''
Извлечение "отрывка" стихотворения:

- Начинаем с найденной строки (start_idx).
- Поднимаемся вверх по строкам, пока не встретим пустую строку (или начало файла) — так находим начало стихотворения.
- Затем идём вниз, собирая строки, пока не встретим строку, заканчивающуюся на ., !, ? или ....
- Возвращаем список строк — это и есть "смысловой отрывок".
'''

def find_semantic_passage(query):
    words = query.lower().split()
    vectors = [w2v_model.wv[w] for w in words if w in w2v_model.wv]
    if not vectors:
        return None
    query_vec = np.mean(vectors, axis=0).reshape(1, -1)
    sims = cosine_similarity(query_vec, corpus_vectors)[0]
    best_idx = sims.argmax()
    passage = extract_passage(best_idx)
    return "\n".join(passage)

'''
Поиск наиболее семантически близкого отрывка к запросу:

- Разбиваем запрос на слова.
- Находим векторы слов, которые есть в модели Word2Vec.
- Усредняем векторы — получаем вектор запроса.
- Сравниваем этот вектор с векторами всех строк корпуса (через косинусную близость).
- Находим индекс строки с наибольшей схожестью.
- Извлекаем отрывок, начиная с этой строки.
- Возвращаем отрывок как строку с переносами.
'''

def create_text_image(text: str, width=800, padding=20):
    import random
    from PIL import Image, ImageDraw, ImageFont
    import io

    bg_filename = f"{random.choice(['1','2','3','4'])}.jpg" 
    bg_img = Image.open(bg_filename).convert("RGBA")

    wpercent = width / bg_img.size[0]
    hsize = int(bg_img.size[1] * wpercent)
    bg_img = bg_img.resize((width, hsize), Image.Resampling.LANCZOS)

    draw = ImageDraw.Draw(bg_img)
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    lines = []
    
    paragraphs = text.split("\n")

    for para in paragraphs:
        words = para.split()
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            bbox = draw.textbbox((0, 0), test_line, font=font)
            w = bbox[2] - bbox[0]
            if w + 2 * padding > width:
                lines.append(line)
                line = word
            else:
                line = test_line
        if line:
            lines.append(line)
        lines.append("")

    if lines and lines[-1] == "":
        lines.pop()

    bbox = font.getbbox("A")
    line_height = (bbox[3] - bbox[1]) + 8

    total_text_height = line_height * len(lines)
    y = (bg_img.size[1] - total_text_height) // 2

    rectangle_height = total_text_height + padding
    rectangle_y = y - padding // 2

    rectangle_x0 = 0  
    rectangle_x1 = width  

    draw.rectangle(
        [(rectangle_x0, rectangle_y), (rectangle_x1, rectangle_y + rectangle_height)],
        fill=(255, 255, 255, 120)
    )

    for line in lines:
        if line.strip() == "":
            y += line_height 
            continue
        bbox = draw.textbbox((0, 0), line, font=font)
        w = bbox[2] - bbox[0]
        draw.text(((width - w) / 2, y), line, fill="black", font=font)
        y += line_height

    bio = io.BytesIO()
    bg_img.save(bio, format="PNG")
    bio.seek(0)
    return bio

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Напишите фразу, а я подберу по смыслу отрывок из стихов.")

chat_last_passage = {}

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text.strip()

    if is_nonsense(query):
        await update.message.reply_text("Не понимаю, напишите по‑другому.")
        return

    result = find_semantic_passage(query)
    if result:
        chat_last_passage[update.effective_chat.id] = result

        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton("Создать открытку", callback_data="create_postcard")]
        ])

        await update.message.reply_text(result, reply_markup=keyboard)
    else:
        await update.message.reply_text("Не понимаю, напишите по‑другому.")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    chat_id = query.message.chat.id

    text = chat_last_passage.get(chat_id)
    if not text:
        await query.message.reply_text("Нет текста для создания открытки. Напишите что-нибудь боту.")
        return

    image_buffer = create_text_image(text)
    await query.message.reply_photo(photo=InputFile(image_buffer, filename="postcard.png"))

def main():
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(button_callback, pattern="create_postcard"))
    print("Бот запущен. Нажмите Ctrl+C для остановки.")
    app.run_polling()

if __name__ == "__main__":
    main()
