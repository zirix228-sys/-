# Устаревший скрипт для сборки небольшого FAISS-индекса на банковском корпусе.
# В текущей версии проекта основной индекс строится через build_corpus_index.py.
# Файл оставлен как резервный вариант для быстрой проверки на малом наборе данных.
from pathlib import Path
from parallel_corpus_faiss import ParallelCorpusFAISS

BASE_DIR = Path(__file__).resolve().parent


def load_corpus(ru_file: Path, en_file: Path):
    with open(ru_file, "r", encoding="utf-8") as f:
        ru_texts = [line.strip() for line in f if line.strip()]

    with open(en_file, "r", encoding="utf-8") as f:
        en_texts = [line.strip() for line in f if line.strip()]

    if len(ru_texts) != len(en_texts):
        raise ValueError(f"Разное количество строк: RU={len(ru_texts)}, EN={len(en_texts)}")

    return ru_texts, en_texts


if __name__ == "__main__":
    ru_path = BASE_DIR / "data" / "filtered_corpus_ru.txt"
    en_path = BASE_DIR / "data" / "filtered_corpus_en.txt"

    save_folder = BASE_DIR / "data" / "faiss_index"
    save_folder.mkdir(parents=True, exist_ok=True)

    save_path = save_folder / "parallel_corpus"

    print("Загружаем банковский корпус...")
    ru_texts, en_texts = load_corpus(ru_path, en_path)

    print(f"Загружено пар: {len(ru_texts)}")

    db = ParallelCorpusFAISS(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        index_type="Flat",
        nlist=1
    )

    db.build_index(
        ru_texts=ru_texts,
        en_texts=en_texts,
        batch_size=128,
        save_path=str(save_path)
    )

    print(f"Готово! Индекс сохранен: {save_path}")