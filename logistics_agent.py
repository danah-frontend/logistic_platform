import os
import re
import uuid
import string
import requests
import pandas as pd
import numpy as np
import urllib3
from datetime import datetime, timedelta
from dotenv import load_dotenv
 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
from pypdf import PdfReader
from langchain_gigachat import GigaChat
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_classic.memory import ConversationBufferMemory
 
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
 
# ─────────────────────────── ENV & LLM SETUP ───────────────────────────────
load_dotenv(dotenv_path="key.env")
GigaChatKey = os.getenv("GIGACHAT_API_KEY")
 
llm = GigaChat(
    credentials=GigaChatKey,
    scope="GIGACHAT_API_PERS",
    model="GigaChat",
    temperature=0.5,
    verify_ssl_certs=False,
    max_tokens=1500,
)
 
# ─────────────────────────── SALUTE SPEECH TOKEN ───────────────────────────
load_dotenv(dotenv_path="salut.env")
auth_token = os.getenv("SALUT")
salute_token = ""
 
try:
    rq_uid = str(uuid.uuid4())
    url_oauth = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    headers_oauth = {
        "Content-Type": "application/x-www-form-urlencoded",
        "RqUID": rq_uid,
        "Authorization": f"Basic {auth_token}",
    }
    resp = requests.post(url_oauth, headers=headers_oauth, data={"scope": "SALUTE_SPEECH_PERS"}, verify=False)
    salute_token = resp.json().get("access_token", "")
except Exception as e:
    print(f"[WARN] Salute Speech token error: {e}")
 
# ─────────────────────────── MEMORY ────────────────────────────────────────
dispatcher_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output",
)
analyst_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output",
)
 
# ─────────────────────────── KNOWLEDGE BASE (RAG) ──────────────────────────
RULES_DIR = "files"
rag_docs: list[Document] = []
 
for fname in ["perevozki.pdf", "adr.pdf"]:
    fpath = os.path.join(RULES_DIR, fname)
    if os.path.exists(fpath):
        try:
            reader = PdfReader(fpath)
            for pnum, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    rag_docs.append(Document(page_content=text, metadata={"source": fname, "page": pnum}))
        except:
            pass
 
for fname in ["tariffs.txt", "routes.txt", "cargo_rules.txt"]:
    fpath = os.path.join(RULES_DIR, fname)
    if os.path.exists(fpath):
        with open(fpath, encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            rag_docs.append(Document(page_content=content, metadata={"source": fname}))
 
if not rag_docs:
    builtin_rules = """
ПРАВИЛА ПЕРЕВОЗКИ ГРУЗОВ (встроенная база)
 
1. ОПАСНЫЕ ГРУЗЫ (ADR)
Перевозка опасных грузов регулируется ДОПОГ (ADR). Требуется специальный допуск водителя.
Классы опасности: 1-9.
 
2. ТЕМПЕРАТУРНЫЙ РЕЖИМ
Продукты питания (мясо, рыба, молочка): 0 до +5°C. Заморозка: -18°C и ниже.
 
3. ДОКУМЕНТАЦИЯ
ТТН, путевой лист, CMR для международных перевозок.
"""
    rag_docs.append(Document(page_content=builtin_rules, metadata={"source": "builtin_rules"}))
 
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=500, chunk_overlap=100)
split_docs = text_splitter.split_documents(rag_docs)
 
def tokenize(s: str) -> list[str]:
    return s.lower().translate(str.maketrans("", "", string.punctuation)).split()
 
bm25_retriever = BM25Retriever.from_documents(documents=split_docs, preprocess_func=tokenize, k=4)
 
# ─────────────────────────── ML MODEL ──────────────────────────────────────
CARGO_CLASSES = {
    "продукт": 0, "яблок": 0, "овощ": 0, "фрукт": 0, "молок": 0, "мяс": 0,
    "рыб": 0, "птиц": 0, "медикамент": 4, "препарат": 4, "лекарств": 4,
    "электроник": 1, "техник": 1, "оборудован": 1, "компьютер": 1,
    "стройматериал": 2, "кирпич": 2, "песок": 2, "цемент": 2, "бетон": 2,
    "хими": 3, "краск": 3, "раствор": 3, "бензин": 3,
    "мебел": 4, "одежд": 4, "текстил": 4, "мед": 4, "медь": 4, "металл": 4,
}
 
def classify_cargo(cargo_name: str) -> str:
    name_lower = cargo_name.lower()
    for keyword, cls in CARGO_CLASSES.items():
        if keyword in name_lower:
            return ["perishable", "electronics", "construction", "chemicals", "general"][cls]
    return "general"
 
# ─────────────────────────── ОБУЧАЮЩАЯ ВЫБОРКА (train) ─────────────────────
# 30 примеров: 10 срочных, 10 стандартных, 10 плановых
# Признаки: [вес_кг, дней_до_срока, тип_груза_id, расстояние_км]
# Типы груза: 0=скоропорт, 1=электроника, 2=стройматериалы, 3=химикаты, 4=общий
 
_train_X = np.array([
    # ── СРОЧНЫЕ (класс 2): мало дней, часто небольшой вес ──────────────────
    [160,   0,  4,  1380],  # медикаменты Сочи→Рязань, сегодня
    [160,   1,  4,  1380],  # медикаменты, завтра
    [100,   0,  0,    50],  # скоропорт, сегодня, близко
    [200,   1,  1,   100],  # электроника, завтра
    [ 50,   0,  1,    30],  # мелкая электроника, сегодня
    [300,   1,  4,   100],  # общий груз, завтра
    [500,   2,  0,   200],  # скоропорт, 2 дня
    [250,   1,  3,   400],  # химикаты, завтра — всегда срочно
    [ 80,   0,  0,   150],  # молочная продукция, сегодня
    [400,   2,  1,   300],  # электроника, 2 дня, срочный заказ
 
    # ── СТАНДАРТНЫЕ (класс 1): средний вес, 3–8 дней ───────────────────────
    [1000,  5,  0,   400],  # продукты, 5 дней
    [3000,  7,  4,   600],  # общий, неделя
    [2000,  3,  0,   300],  # скоропорт, 3 дня
    [4000,  8,  2,   500],  # стройматериалы, 8 дней
    [ 900,  4,  0,   250],  # продукты, 4 дня
    [3500,  4,  1,   500],  # электроника, 4 дня
    [1500,  6,  4,   700],  # общий, 6 дней
    [2500,  5,  3,   350],  # химикаты, 5 дней
    [1200,  3,  2,   420],  # стройматериалы, 3 дня
    [2800,  7,  1,   550],  # техника, неделя
 
    # ── ПЛАНОВЫЕ (класс 0): большой вес, много дней ─────────────────────────
    [ 5000, 10,  2,   800],  # стройматериалы, 10 дней
    [20000, 14,  2,  1000],  # крупная партия, 2 недели
    [ 8000, 12,  3,   700],  # химикаты оптом, 12 дней
    [15000, 20,  2,  1200],  # стройматериалы, 3 недели
    [10000, 10,  2,  1500],  # стройматериалы дальний маршрут
    [25000, 15,  2,  2000],  # очень крупная партия
    [12000, 11,  4,   900],  # общий крупный груз
    [ 7000, 14,  3,  1100],  # химикаты, плановая поставка
    [18000, 18,  2,  1600],  # стройматериалы
    [ 9000, 13,  4,   950],  # общий, большой срок
])
 
_train_y = np.array([
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,   # 10 срочных
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1,   # 10 стандартных
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,   # 10 плановых
])
 
# ─────────────────────────── ТЕСТОВАЯ ВЫБОРКА (test) ────────────────────────
# 12 примеров: 4 срочных, 4 стандартных, 4 плановых
# ВАЖНО: тестовые примеры НЕ использовались при обучении —
# только на них можно честно оценить качество модели
 
_test_X = np.array([
    # ── СРОЧНЫЕ (класс 2) ───────────────────────────────────────────────────
    [120,   1,  0,   180],  # скоропорт, завтра, близко
    [350,   2,  4,   250],  # общий, 2 дня, срочный
    [ 90,   0,  1,    90],  # электроника, сегодня
    [200,   1,  3,   500],  # химикаты, завтра
 
    # ── СТАНДАРТНЫЕ (класс 1) ───────────────────────────────────────────────
    [1800,  5,  4,   450],  # общий, 5 дней
    [2200,  6,  0,   380],  # скоропорт, 6 дней
    [3200,  4,  2,   600],  # стройматериалы, 4 дня
    [1100,  7,  1,   420],  # электроника, неделя
 
    # ── ПЛАНОВЫЕ (класс 0) ──────────────────────────────────────────────────
    [11000, 12,  2,  1050],  # стройматериалы, 12 дней
    [16000, 16,  4,  1300],  # общий крупный, 16 дней
    [ 6500, 10,  3,   850],  # химикаты, 10 дней
    [22000, 19,  2,  1800],  # крупная партия, 19 дней
])
 
_test_y = np.array([
    2, 2, 2, 2,   # 4 срочных
    1, 1, 1, 1,   # 4 стандартных
    0, 0, 0, 0,   # 4 плановых
])
 
# ─────────────────────────── ОБУЧЕНИЕ МОДЕЛИ ────────────────────────────────
priority_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
priority_model.fit(_train_X, _train_y)
 
# ─────────────────────────── ОЦЕНКА КАЧЕСТВА МОДЕЛИ ────────────────────────
_label_names = ["Плановая", "Стандартная", "Срочная"]
_feature_names = ["вес_кг", "дней_до_срока", "тип_груза_id", "расстояние_км"]
 
print("\n" + "="*60)
print("ОЦЕНКА КАЧЕСТВА ML-МОДЕЛИ")
print("="*60)
 
# 1. Точность на обучающей выборке
_y_pred_train = priority_model.predict(_train_X)
_train_acc = accuracy_score(_train_y, _y_pred_train)
print(f"\n[1] Точность на обучающей выборке (30 примеров): {_train_acc:.2%}")
print("    Если 100% — модель идеально запомнила обучающие данные.")
print("    Это ожидаемо для RandomForest, но не говорит о реальном качестве.\n")
 
# 2. Точность на тестовой выборке — самая честная оценка
_y_pred_test = priority_model.predict(_test_X)
_test_acc = accuracy_score(_test_y, _y_pred_test)
print(f"[2] Точность на тестовой выборке (12 примеров): {_test_acc:.2%}")
print("    Тестовые данные модель не видела — это честная оценка.\n")
 
# 3. Подробный отчёт по тестовой выборке
print("[3] Отчёт по классам (тестовая выборка):")
print(classification_report(_test_y, _y_pred_test, target_names=_label_names))
 
# 4. Матрица ошибок на тестовой выборке
_cm_test = confusion_matrix(_test_y, _y_pred_test)
print("[4] Матрица ошибок — тестовая выборка")
print("    (строки = правильный класс, столбцы = предсказанный класс)")
print(f"    {'':15} {'Плановая':>12} {'Стандартная':>12} {'Срочная':>10}")
for i, label in enumerate(_label_names):
    row = _cm_test[i]
    # помечаем диагональ (правильные) символом ✓
    cells = []
    for j, val in enumerate(row):
        mark = f"{val}✓" if i == j else str(val)
        cells.append(mark)
    print(f"    {label:15} {cells[0]:>12} {cells[1]:>12} {cells[2]:>10}")
print()
 
# 5. Кросс-валидация по всей выборке (train+test объединяем)
_all_X = np.vstack([_train_X, _test_X])
_all_y = np.concatenate([_train_y, _test_y])
_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
_cv_scores = cross_val_score(priority_model, _all_X, _all_y, cv=_cv, scoring="f1_macro")
print(f"[5] Кросс-валидация (3-fold, F1 macro, вся выборка 42 примера):")
print(f"    Фолды: {[f'{s:.2%}' for s in _cv_scores]}")
print(f"    Среднее F1: {_cv_scores.mean():.2%} ± {_cv_scores.std():.2%}\n")
 
# 6. Важность признаков
_importances = priority_model.feature_importances_
print("[6] Важность признаков:")
for name, imp in sorted(zip(_feature_names, _importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 40)
    print(f"    {name:20} {imp:.3f}  {bar}")
 
print("\n" + "="*60)
print(f"ИТОГ: train accuracy={_train_acc:.2%} | test accuracy={_test_acc:.2%}")
print(f"      CV F1={_cv_scores.mean():.2%} ± {_cv_scores.std():.2%}")
if _test_acc >= 0.75:
    print("      Модель работает приемлемо для учебного проекта.")
else:
    print("      Качество низкое — нужно расширить обучающую выборку.")
print("="*60 + "\n")
 
PRIORITY_LABELS = {0: "Плановая", 1: "Стандартная", 2: "Срочная"}
VEHICLE_MAP = {
    "perishable": "Рефрижератор",
    "electronics": "Тентованный (осторожно)",
    "construction": "Самосвал / платформа",
    "chemicals": "Цистерна / спецтранспорт",
    "general": "Тентованный / фургон",
}
 
def predict_priority(weight_kg: float, days_until_deadline: float, cargo_type: str, distance_km: float) -> tuple[str, str]:
    cargo_ids = {"perishable": 0, "electronics": 1, "construction": 2, "chemicals": 3, "general": 4}
    cid = cargo_ids.get(cargo_type, 4)
    X = np.array([[weight_kg, days_until_deadline, cid, distance_km]])
    
    print(f"\n[ML-DEBUG] predict_priority вызван:")
    print(f"  - weight_kg: {weight_kg}")
    print(f"  - days_until_deadline: {days_until_deadline}")
    print(f"  - cargo_type: {cargo_type} (ID: {cid})")
    print(f"  - distance_km: {distance_km}")
    print(f"  - Вектор признаков: {X[0]}")
    
    priority_id = priority_model.predict(X)[0]
    
    print(f"  - Предсказанный ID: {priority_id} -> {PRIORITY_LABELS[priority_id]}")
    
    vehicle = VEHICLE_MAP.get(cargo_type, VEHICLE_MAP["general"])
    return PRIORITY_LABELS[priority_id], vehicle
 
# ─────────────────────────── ORDERS STORAGE ────────────────────────────────
ORDERS_FILE = os.path.join(RULES_DIR, "orders.csv")
os.makedirs(RULES_DIR, exist_ok=True)
 
def _load_orders() -> pd.DataFrame:
    if os.path.exists(ORDERS_FILE):
        try:
            return pd.read_csv(ORDERS_FILE, encoding="utf-8")
        except:
            pass
    return pd.DataFrame(columns=[
        "id", "created_at", "cargo", "weight_kg", "from_city", "to_city",
        "deadline", "distance_km", "cargo_type", "priority", "vehicle", "status", "notes"
    ])
 
def _save_orders(df: pd.DataFrame):
    df.to_csv(ORDERS_FILE, index=False, encoding="utf-8")
 
# ─────────────────────────── CITY DISTANCES ─────────────────────────────────
CITY_DISTANCES = {
    ("москва", "санкт-петербург"): 712, ("москва", "питер"): 712, ("москва", "спб"): 712,
    ("москва", "казань"): 815, ("москва", "нижний новгород"): 411, ("москва", "екатеринбург"): 1784,
    ("москва", "новосибирск"): 3335, ("москва", "тверь"): 167, ("москва", "воронеж"): 524,
    ("москва", "ростов-на-дону"): 1070, ("москва", "краснодар"): 1354, ("москва", "самара"): 1055,
    ("москва", "уфа"): 1341, ("москва", "пермь"): 1386, ("москва", "челябинск"): 1908,
    ("москва", "суздаль"): 230, ("москва", "сочи"): 1620, ("москва", "рязань"): 198,
    ("сочи", "москва"): 1620, ("сочи", "рязань"): 1380, ("рязань", "сочи"): 1380,
    ("тверь", "москва"): 167, ("суздаль", "москва"): 230, ("рязань", "москва"): 198,
}
 
_CITY_NORM = {
    "москвы": "москва", "москву": "москва",
    "петербурга": "санкт-петербург", "санкт-петербурга": "санкт-петербург",
    "питера": "питер", "казани": "казань", "твери": "тверь",
    "сочи": "сочи", "рязани": "рязань", "рязань": "рязань",
}
 
def _normalize_city(raw: str) -> str:
    s = raw.lower().strip()
    s = re.sub(r"^(к|до|через|в город)\s+", "", s)
    return _CITY_NORM.get(s, s)
 
def get_distance(from_city: str, to_city: str) -> int:
    f = _normalize_city(from_city)
    t = _normalize_city(to_city)
    key = (f, t)
    key_rev = (t, f)
    dist = CITY_DISTANCES.get(key) or CITY_DISTANCES.get(key_rev)
    if dist:
        return dist
    return 500
 
def parse_order_text(text: str) -> dict:
    """Извлекает поля заявки из текста"""
    result = {
        "cargo": "",
        "weight_kg": 0.0,
        "from_city": "",
        "to_city": "",
        "deadline": "",
        "notes": "",
    }
    
    print(f"\n[DEBUG] Парсим: {text}")
    
    # 1. ПОИСК ВЕСА
    weight_found = False
    w = re.search(r"(\d+(?:[.,]\d+)?)\s*(кг|кило|тонн|т)\b", text, re.IGNORECASE)
    if w:
        val = float(w.group(1).replace(",", "."))
        if w.group(2).lower() in ["тонн", "т"]:
            val *= 1000
        result["weight_kg"] = val
        weight_found = True
        print(f"[DEBUG] Вес: {val} кг")
    
    if not weight_found:
        numbers = re.findall(r"\b(\d{2,5})\b", text)
        for num in numbers:
            val = int(num)
            if 50 <= val <= 50000:
                result["weight_kg"] = val
                print(f"[DEBUG] Вес (без единиц): {val} кг")
                break
    
    # 2. ПОИСК ГОРОДА ОТПРАВЛЕНИЯ
    from_match = re.search(r"из\s+([А-ЯЁ][а-яё]+)", text, re.IGNORECASE)
    if from_match:
        city_raw = from_match.group(1).strip()
        city_raw = re.sub(r"\s+(доставить|в|до|к|,).*$", "", city_raw, flags=re.IGNORECASE)
        result["from_city"] = city_raw.capitalize()
        print(f"[DEBUG] Откуда: {result['from_city']}")
    
    # 3. ПОИСК ГОРОДА НАЗНАЧЕНИЯ
    to_patterns = [
        r"доставить\s+в\s+([А-ЯЁ][а-яё]+)",
        r"\s+в\s+([А-ЯЁ][а-яё]+)(?=\s|,|$)",
        r"до\s+([А-ЯЁ][а-яё]+)",
    ]
    to_match = None
    for pattern in to_patterns:
        to_match = re.search(pattern, text, re.IGNORECASE)
        if to_match:
            break
    
    if to_match:
        city_raw = to_match.group(1).strip()
        city_raw = re.sub(r"\s+к\s+.*$", "", city_raw, flags=re.IGNORECASE)
        result["to_city"] = city_raw.capitalize()
        print(f"[DEBUG] Куда: {result['to_city']}")
    
    # 4. ПОИСК ГРУЗА
    cargo_match = re.search(r"(?:забрать|доставить|перевезти)\s+(\d*\s*кг?\s*)?([А-ЯЁа-яё\s]{2,30}?)(?=\s+из|\s+в|\s+до|$)", text, re.IGNORECASE)
    if cargo_match:
        cargo_raw = cargo_match.group(2) if cargo_match.lastindex >= 2 else cargo_match.group(1)
        if cargo_raw:
            cargo_raw = re.sub(r"\d+\s*кг?", "", cargo_raw).strip()
            if len(cargo_raw) > 2:
                result["cargo"] = cargo_raw
                print(f"[DEBUG] Груз: {result['cargo']}")
    
    if not result["cargo"]:
        words = text.split()
        for word in words[:5]:
            if word.lower() not in ["срочно", "забрать", "доставить", "перевезти", "из", "в", "до", "к", "кг"]:
                if len(word) > 3:
                    result["cargo"] = word
                    print(f"[DEBUG] Груз (по умолчанию): {result['cargo']}")
                    break
    
    # 5. ПОИСК СРОКА
    deadline_match = re.search(r"(?:к|до)\s+(\d{1,2})\s+(мая|июня|июля|августа|сентября|октября|ноября|декабря|января|февраля|марта|апреля)", text, re.IGNORECASE)
    if deadline_match:
        result["deadline"] = f"{deadline_match.group(1)} {deadline_match.group(2)}"
        print(f"[DEBUG] Срок: {result['deadline']}")
    else:
        day_match = re.search(r"к\s+(понедельнику|вторнику|среде|четвергу|пятнице|субботе|воскресенью)", text, re.IGNORECASE)
        if day_match:
            result["deadline"] = day_match.group(1)
            print(f"[DEBUG] Срок (день): {result['deadline']}")
    
    return result
 
# ─────────────────────────── TOOLS ─────────────────────────────────────────
 
@tool
def audio_to_text(input: str) -> str:
    """Распознаёт речь из WAV-файла"""
    print(f"\n[AUDIO_DEBUG] Вызван")
    
    if ".wav" not in input.lower():
        return input
    
    m = re.search(r"([A-Za-zА-Яа-я]:[^\s]+\.wav|/[^\s]+\.wav)", input)
    if not m:
        return "Не удалось найти путь к аудиофайлу"
    
    audio_path = m.group(1)
    
    if not os.path.exists(audio_path):
        return f"Файл не найден: {audio_path}"
    
    if not salute_token:
        return "Голосовое распознавание не настроено"
    
    url_rec = "https://smartspeech.sber.ru/rest/v1/speech:recognize"
    headers = {
        "Authorization": f"Bearer {salute_token}",
        "Content-Type": "audio/x-pcm;bit=16;rate=16000",
    }
    
    with open(audio_path, "rb") as f:
        data = f.read()
    
    try:
        resp = requests.post(url_rec, headers=headers, data=data, verify=False)
        print(f"[AUDIO_DEBUG] Статус: {resp.status_code}")
        
        if resp.status_code == 200:
            result = resp.json()
            print(f"[AUDIO_DEBUG] Результат: {result}")
            
            # Извлекаем текст
            if isinstance(result.get("result"), list):
                recognized = result["result"][0] if result["result"] else ""
            else:
                recognized = result.get("result", "")
            
            # Очищаем от лишних пробелов и точек
            recognized = recognized.strip().strip('.')
            
            if recognized:
                print(f"[AUDIO_DEBUG] Успешно распознано: {recognized}")
                return recognized
            else:
                return "Не удалось распознать речь (пустой результат)"
        else:
            return f"Ошибка {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        print(f"[AUDIO_DEBUG] Ошибка: {e}")
        return f"Ошибка соединения: {e}"
 
@tool
def text_to_audio(text: str) -> str:
    """Синтезирует голосовой ответ"""
    if not salute_token:
        return ""
    url_synth = "https://smartspeech.sber.ru/rest/v1/text:synthesize"
    headers = {"Authorization": f"Bearer {salute_token}", "Content-Type": "application/text"}
    params = {"format": "wav16", "voice": "Bys_24000"}
    try:
        resp = requests.post(url_synth, headers=headers, params=params, data=text.encode("utf-8"), verify=False)
        if resp.status_code == 200:
            out_path = os.path.join(RULES_DIR, "response.wav")
            with open(out_path, "wb") as f:
                f.write(resp.content)
            return out_path
        return ""
    except:
        return ""
 
@tool
def create_order(text: str) -> str:
    """Создаёт заявку на перевозку"""
    print(f"\n{'='*60}")
    print(f"[CREATE_ORDER] Входной текст: {text}")
    print(f"{'='*60}")
    
    parsed = parse_order_text(text)
    
    cargo = parsed["cargo"] if parsed["cargo"] else "Не указано"
    weight = parsed["weight_kg"] if parsed["weight_kg"] > 0 else 1000.0
    from_city = parsed["from_city"] if parsed["from_city"] else "Москва"
    to_city = parsed["to_city"] if parsed["to_city"] else "Санкт-Петербург"
    deadline = parsed["deadline"] if parsed["deadline"] else "Не указан"
    
    distance = get_distance(from_city, to_city)
    cargo_type = classify_cargo(cargo)
    
    # Расчет дней до дедлайна
    days = 5
    if deadline != "Не указан":
        day_map = {
            "понедельнику": 0, "вторнику": 1, "среде": 2, "четвергу": 3,
            "пятнице": 4, "субботе": 5, "воскресенью": 6,
        }
        for dname, dnum in day_map.items():
            if dname in deadline.lower():
                today = datetime.now().weekday()
                days = (dnum - today) % 7
                if days == 0:
                    days = 7
                break
    
    priority, vehicle = predict_priority(weight, days, cargo_type, distance)
    
    order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    new_row = {
        "id": order_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cargo": cargo,
        "weight_kg": weight,
        "from_city": from_city,
        "to_city": to_city,
        "deadline": deadline,
        "distance_km": distance,
        "cargo_type": cargo_type,
        "priority": priority,
        "vehicle": vehicle,
        "status": "Новая",
        "notes": text[:200],
    }
    
    df = _load_orders()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_orders(df)
    
    print(f"[CREATE_ORDER] СОЗДАНА ЗАЯВКА:")
    print(f"  - Груз: {cargo}")
    print(f"  - Вес: {weight} кг")
    print(f"  - Откуда: {from_city}")
    print(f"  - Куда: {to_city}")
    print(f"  - Расстояние: {distance} км")
    print(f"  - Срок: {deadline}")
    print(f"  - Приоритет: {priority}")
    print(f"  - Транспорт: {vehicle}")
    
    return (
        f"Заявка принята!\n"
        f"Номер: {order_id}\n"
        f"Груз: {cargo}, {weight:.0f} кг\n"
        f"Маршрут: {from_city} → {to_city} (~{distance} км)\n"
        f"Срок: {deadline}\n"
        f"Приоритет: {priority}\n"
        f"Транспорт: {vehicle}\n"
        f"Тип груза: {cargo_type}"
    )
 
@tool
def get_orders_summary(query: str) -> str:
    """Возвращает список заявок"""
    df = _load_orders()
    if df.empty:
        return "Заявок пока нет."
    total = len(df)
    new_cnt = len(df[df["status"] == "Новая"])
    summary = f"Всего заявок: {total}, новых: {new_cnt}\n\n"
    last = df.tail(5)
    for _, row in last.iterrows():
        summary += f"[{row['id']}] {row['cargo']} | {row['from_city']}→{row['to_city']} | {row['weight_kg']} кг | {row['priority']}\n"
    return summary
 
@tool
def consult_logistics_rules(query: str) -> str:
    """Консультация по правилам перевозки"""
    docs = bm25_retriever.invoke(query)
    context = "\n\n".join(d.page_content for d in docs)
    
    chain = PromptTemplate.from_template(
        "Ты эксперт по логистике. Ответь на вопрос, используя контекст. Будь краток.\n\n"
        "Контекст:\n{context}\n\nВопрос: {query}\n\nОтвет:"
    ) | llm
    return chain.invoke({"context": context, "query": query}).content
 
@tool
def estimate_cost(input: str) -> str:
    """Оценивает стоимость перевозки"""
    parsed = parse_order_text(input)
    from_city = parsed["from_city"] or "Москва"
    to_city = parsed["to_city"] or "Санкт-Петербург"
    weight = parsed["weight_kg"] if parsed["weight_kg"] > 0 else 1000
    
    distance = get_distance(from_city, to_city)
    tons = weight / 1000
    base_cost = 50 * tons * distance
    if base_cost < 15000:
        base_cost = 15000
    
    cargo_type = classify_cargo(parsed["cargo"])
    multiplier = {"perishable": 1.3, "chemicals": 1.6, "electronics": 1.2}.get(cargo_type, 1.0)
    final = base_cost * multiplier
    
    return (
        f"Оценка стоимости:\n"
        f"Маршрут: {from_city} → {to_city} (~{distance} км)\n"
        f"Вес: {weight:.0f} кг ({tons:.2f} т)\n"
        f"Итого: ~{final:,.0f} ₽"
    )
 
# ─────────────────────────── AGENTS ────────────────────────────────────────
dispatcher_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """Ты диспетчер логистики. ОБЯЗАТЕЛЬНО используй инструменты.
 
Если пользователь описывает перевозку - вызови create_order с полным текстом пользователя.
Если спрашивают стоимость - вызови estimate_cost.
Если спрашивают список заявок - вызови get_orders_summary.
Никогда не отвечай сам, всегда вызывай соответствующий инструмент!"""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
 
dispatcher_tools = [create_order, estimate_cost, get_orders_summary, audio_to_text, text_to_audio]
dispatcher_agent = create_tool_calling_agent(llm, dispatcher_tools, dispatcher_prompt)
dispatcher_executor = AgentExecutor(
    agent=dispatcher_agent,
    tools=dispatcher_tools,
    verbose=True,
    memory=dispatcher_memory,
    max_iterations=8,
    handle_parsing_errors=True,
)
 
analyst_prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты аналитик по логистике. Всегда вызывай consult_logistics_rules для ответа на вопросы."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
 
analyst_tools = [consult_logistics_rules, estimate_cost, get_orders_summary]
analyst_agent = create_tool_calling_agent(llm, analyst_tools, analyst_prompt)
analyst_executor = AgentExecutor(
    agent=analyst_agent,
    tools=analyst_tools,
    verbose=True,
    memory=analyst_memory,
    max_iterations=5,
    handle_parsing_errors=True,
)
 
# ─────────────────────────── PUBLIC API ─────────────────────────────────────
def load_orders() -> pd.DataFrame:
    return _load_orders()
 
def get_audio_response_path() -> str:
    p = os.path.join(RULES_DIR, "response.wav")
    return p if os.path.exists(p) else ""
 
# ─────────────────────────── DIRECT ORDER CREATION ─────────────────────────
def create_order_direct(text: str) -> str:
    """Прямое создание заявки без агента (с отладкой)"""
    print(f"\n[DIRECT] Создаю заявку: {text}")
    
    parsed = parse_order_text(text)
    
    cargo = parsed["cargo"] if parsed["cargo"] else "Не указано"
    weight = parsed["weight_kg"] if parsed["weight_kg"] > 0 else 1000.0
    from_city = parsed["from_city"] if parsed["from_city"] else "Москва"
    to_city = parsed["to_city"] if parsed["to_city"] else "Санкт-Петербург"
    deadline = parsed["deadline"] if parsed["deadline"] else "Не указан"
    
    distance = get_distance(from_city, to_city)
    cargo_type = classify_cargo(cargo)
    
    days = 5
    print(f"\n[ML-DEBUG] Анализ дедлайна: '{deadline}'")
    
    if deadline and deadline != "Не указан":
        if "сегодня" in deadline.lower():
            days = 0
            print(f"[ML-DEBUG] Дней: {days} (сегодня)")
        elif "завтра" in deadline.lower():
            days = 1
            print(f"[ML-DEBUG] Дней: {days} (завтра)")
        else:
            day_map = {
                "понедельник": 0, "понедельнику": 0,
                "вторник": 1, "вторнику": 1,
                "среда": 2, "среде": 2,
                "четверг": 3, "четвергу": 3,
                "пятница": 4, "пятнице": 4,
                "суббота": 5, "субботе": 5,
                "воскресенье": 6, "воскресенью": 6,
            }
            today = datetime.now().weekday()
            found = False
            
            for day_name, day_num in day_map.items():
                if day_name in deadline.lower():
                    days = (day_num - today) % 7
                    if days == 0:
                        days = 7
                    found = True
                    print(f"[ML-DEBUG] День недели: {day_name} (номер {day_num}), сегодня {today}, дней до: {days}")
                    break
            
            if not found:
                date_match = re.search(r"(\d{1,2})\s+(мая|июня|июля|августа|сентября|октября|ноября|декабря|января|февраля|марта|апреля)", deadline.lower())
                if date_match:
                    day = int(date_match.group(1))
                    month_name = date_match.group(2)
                    months = {
                        "января": 1, "февраля": 2, "марта": 3, "апреля": 4,
                        "мая": 5, "июня": 6, "июля": 7, "августа": 8,
                        "сентября": 9, "октября": 10, "ноября": 11, "декабря": 12
                    }
                    month = months.get(month_name, 1)
                    current_year = datetime.now().year
                    target_date = datetime(current_year, month, day)
                    if target_date < datetime.now():
                        target_date = datetime(current_year + 1, month, day)
                    days = (target_date - datetime.now()).days
                    print(f"[ML-DEBUG] Дата: {day}.{month}, целевая дата: {target_date.date()}, дней до: {days}")
    
    days = max(0, days)
    print(f"[ML-DEBUG] ИТОГОВОЕ количество дней до дедлайна: {days}")
    
    priority, vehicle = predict_priority(weight, days, cargo_type, distance)
    
    order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    new_row = {
        "id": order_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "cargo": cargo,
        "weight_kg": weight,
        "from_city": from_city,
        "to_city": to_city,
        "deadline": deadline,
        "distance_km": distance,
        "cargo_type": cargo_type,
        "priority": priority,
        "vehicle": vehicle,
        "status": "Новая",
        "notes": text[:200],
    }
    
    df = _load_orders()
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _save_orders(df)
    
    print(f"[DIRECT] СОЗДАНА ЗАЯВКА: вес={weight}, от={from_city}, до={to_city}, расстояние={distance}, приоритет={priority}")
    
    return (
        f"Заявка принята!\n"
        f"Номер: {order_id}\n"
        f"Груз: {cargo}, {weight:.0f} кг\n"
        f"Маршрут: {from_city} → {to_city} (~{distance} км)\n"
        f"Срок: {deadline}\n"
        f"Приоритет: {priority}\n"
        f"Транспорт: {vehicle}\n"
        f"Тип груза: {cargo_type}"
    )
 
 
# ─────────────────────────── VALIDATION FUNCTION ─────────────────────────────
def is_valid_order_text(text: str) -> tuple[bool, str]:
    """Проверяет, является ли текст валидной заявкой на перевозку с помощью GigaChat"""
    
    if not text or len(text.strip()) < 5:
        return False, "Описание заявки слишком короткое."
    
    # Промпт для GigaChat
    validation_prompt = PromptTemplate.from_template(
        """Ты - система проверки заявок на перевозку грузов. Определи, является ли следующий текст ЗАЯВКОЙ НА ПЕРЕВОЗКУ ГРУЗА.
 
Заявка на перевозку груза должна содержать:
1. Действие (забрать, доставить, перевезти, отправить)
2. Груз (что везём)
3. Город отправления (откуда, из)
4. Город назначения (куда, в, до)
 
Примеры ЗАЯВОК:
- "Забрать 500 кг яблок из Твери в Москву"
- "Доставить 2 тонны цемента из Подольска в Санкт-Петербург"
- "Перевезти 160 кг медикаментов из Сочи в Рязань"
 
Примеры НЕ ЗАЯВОК:
- "Привет, как дела"
- "Что сегодня делала"
- "Расскажи о погоде"
- "Спасибо за помощь"
- "Какие документы нужны"
 
Текст для проверки: "{text}"
 
Ответь ТОЛЬКО в формате JSON:
{"is_order": true/false, "reason": "причина ответа"}
 
Не добавляй никакой другой текст, только JSON.
"""
    )
    
    chain = validation_prompt | llm
    
    try:
        response = chain.invoke({"text": text})
        result_text = response.content.strip()
        
        # Извлекаем JSON из ответа
        import json
        json_match = re.search(r'\{[^{}]*\}', result_text)
        if json_match:
            result = json.loads(json_match.group())
            is_order = result.get("is_order", False)
            reason = result.get("reason", "Не удалось определить")
            
            if is_order:
                return True, "OK"
            else:
                return False, reason
        else:
            # Если JSON не найден, пробуем распарсить текстовый ответ
            if "true" in result_text.lower() or "является" in result_text.lower():
                return True, "OK"
            else:
                return False, "Текст не является заявкой на перевозку груза"
                
    except Exception as e:
        print(f"[ERROR] Ошибка при проверке заявки через GigaChat: {e}")
        # Fallback на простую проверку
        return fallback_validation(text)
 
 
def fallback_validation(text: str) -> tuple[bool, str]:
    """Простая проверка заявки (резервная, если GigaChat недоступен)"""
    text_lower = text.lower()
    
    has_action = any(word in text_lower for word in ["забрать", "доставить", "перевезти", "отправить"])
    has_from = "из" in text_lower
    has_to = any(word in text_lower for word in [" в ", " до ", "в город"])
    
    if not has_action:
        return False, "В тексте нет действия (забрать, доставить, перевезти)"
    if not has_from:
        return False, "Не указан город отправления (используйте 'из')"
    if not has_to:
        return False, "Не указан город назначения (используйте 'в' или 'до')"
    
    return True, "OK"
if __name__ == "__main__":
    print("=== Тест диспетчера ===")
    r = dispatcher_executor.invoke({"input": "Забрать 160 кг медицинских препаратов из Сочи, доставить в Рязань к 29 мая"})
    print(r["output"])