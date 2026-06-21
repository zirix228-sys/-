# Скрипт для автоматической оценки качества перевода на тестовых TXT-файлах.
# В текущей версии проекта основное тестирование выполняется через веб-интерфейс,
# однако данный модуль может использоваться для пакетного запуска заранее подготовленных тестов.

import time
from pathlib import Path
import pandas as pd
import sacrebleu
from pipeline import DocumentPipeline

# Набор тестовых файлов для пакетной оценки качества перевода
TESTS = [
    ("RU_EN_small", "tests/data/ru_en/small_source.txt", "tests/data/ru_en/small_reference.txt", "en"),
    ("RU_EN_medium", "tests/data/ru_en/medium_source.txt", "tests/data/ru_en/medium_reference.txt", "en"),
    ("RU_EN_large", "tests/data/ru_en/large_source.txt", "tests/data/ru_en/large_reference.txt", "en"),

    ("EN_RU_small", "tests/data/en_ru/small_source.txt", "tests/data/en_ru/small_reference.txt", "ru"),
    ("EN_RU_medium", "tests/data/en_ru/medium_source.txt", "tests/data/en_ru/medium_reference.txt", "ru"),
    ("EN_RU_large", "tests/data/en_ru/large_source.txt", "tests/data/en_ru/large_reference.txt", "ru"),
]


def read_text(path):
    return Path(path).read_text(encoding="utf-8")

# Расчет BLEU и chrF относительно эталонного перевода
def calculate_metrics(candidate, reference):
    bleu = sacrebleu.corpus_bleu([candidate], [[reference]]).score
    chrf = sacrebleu.corpus_chrf([candidate], [[reference]]).score
    return round(bleu, 2), round(chrf, 2)

# Последовательный запуск всех тестов и сохранение результатов
def main():
    pipeline = DocumentPipeline()
    results = []

    for test_name, source_path, reference_path, target_lang in TESTS:
        print(f"\nЗапускаю тест: {test_name}")

        source_text = read_text(source_path)
        reference_text = read_text(reference_path)

        start = time.time()

        ocr_result = pipeline.run_ocr(
            file_bytes=source_text.encode("utf-8"),
            filename=Path(source_path).name,
            target_lang=target_lang,
        )

        translate_result = pipeline.run_translate(ocr_result)
        final_result = pipeline.run_postprocess(translate_result)

        translated_text = final_result.get("cleaned_text", "")
        elapsed = round(time.time() - start, 2)

        bleu, chrf = calculate_metrics(translated_text, reference_text)

        results.append({
            "test_name": test_name,
            "target_lang": target_lang,
            "source_chars": len(source_text),
            "reference_chars": len(reference_text),
            "translated_chars": len(translated_text),
            "time_sec": elapsed,
            "BLEU": bleu,
            "chrF": chrf,
        })

    df = pd.DataFrame(results)
    Path("tests/results").mkdir(parents=True, exist_ok=True)
    df.to_csv("tests/results/translation_metrics.csv", index=False, encoding="utf-8-sig")

    print("\nГотово. Результаты сохранены в tests/results/translation_metrics.csv")
    print(df)

if __name__ == "__main__":
    main()