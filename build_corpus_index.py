from pathlib import Path
from parallel_corpus_faiss import ParallelCorpusFAISS

BASE_DIR = Path(__file__).resolve().parent

# Загрузка русской и английской частей параллельного корпуса
def load_corpus(ru_file: Path, en_file: Path):
    if not ru_file.exists() or not en_file.exists():
        raise FileNotFoundError(f"Файлы не найдены:\n{ru_file}\n{en_file}")

    with open(ru_file, "r", encoding="utf-8") as f:
        ru_texts = [line.strip() for line in f if line.strip()]

    with open(en_file, "r", encoding="utf-8") as f:
        en_texts = [line.strip() for line in f if line.strip()]

    if len(ru_texts) != len(en_texts):
        raise ValueError(f"Разное количество строк: RU={len(ru_texts)}, EN={len(en_texts)}")

    return ru_texts, en_texts

# Настройки FAISS для большого параллельного корпуса
if __name__ == "__main__":
    ru_path = BASE_DIR / "data" / "filtered_corpus_ru.txt"
    en_path = BASE_DIR / "data" / "filtered_corpus_en.txt"

    save_folder = BASE_DIR / "data" / "faiss_index"
    save_folder.mkdir(parents=True, exist_ok=True)

    save_path = save_folder / "parallel_corpus"

    print("Загрузка большого корпуса...")
    ru_texts, en_texts = load_corpus(ru_path, en_path)

    print(f"Загружено пар: {len(ru_texts):,}")

    index_type = "IVF"
    n_list = 2000

    print(f"Инициализация FAISS: index_type={index_type}, nlist={n_list}")

    db = ParallelCorpusFAISS(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        index_type=index_type,
        nlist=n_list
    )

    # Построение и сохранение векторного индекса
    db.build_index(
        ru_texts=ru_texts,
        en_texts=en_texts,
        batch_size=512,
        save_path=str(save_path)
    )

    print(f"Готово, индекс сохранен: {save_path}")