import logging
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from ocr.qwen_ocr import QwenOCR
from parallel_corpus_faiss import ParallelCorpusFAISS

logger = logging.getLogger(__name__)

# Основной пайплайн обработки и перевода документов
class DocumentPipeline:
    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        self.config = self._load_config()
        self._setup_logging()
        self.retriever = self._init_retriever()
        model_name = self.config.get("llm", {}).get("model", "qwen3-vl:8b-instruct")
        self.ocr_engine = QwenOCR(model=model_name, retriever=self.retriever)
        logger.info(f"Пайплайн готов. Модель: {model_name}")

# Загрузка параметров модели и настроек RAG
    def _load_config(self) -> Dict[str, Any]:
        config_path = self.config_dir / "main.yaml"
        if not config_path.exists():
            return {
                "llm": {"model": "qwen3-vl:8b-instruct"},
                "rag": {"index_path": "data/faiss_index/parallel_corpus"}
            }
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Ошибка загрузки конфига: {e}")
            return {}

    def _setup_logging(self):
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                handlers=[logging.StreamHandler()]
            )

# Подключение FAISS-базы для поиска переводческих примеров
    def _init_retriever(self) -> Optional[ParallelCorpusFAISS]:
        try:
            # Преобразуем путь из конфига в Path для кроссплатформенности
            index_path_str = self.config.get("rag", {}).get("index_path", "data/faiss_index/parallel_corpus")
            index_path = str(Path(index_path_str)) # Path нормализует слеши
            
            retriever = ParallelCorpusFAISS.load(index_path)
            logger.info(f"RAG-база успешно загружена: {index_path}")
            return retriever
        except Exception as e:
            logger.warning(f"RAG-база не загружена: {e}. Работаем без примеров.")
            return None

    # Запуск распознавания и перевода документа
    def run_ocr(self, file_bytes: bytes, filename: str, target_lang: str = "en", source_lang: Optional[str] = None) -> Dict[str, Any]:
        # Убрали ветвление, просто вызываем Qwen
        logger.info(f"Обработка файла: {filename} | Язык: {target_lang.upper()}")
        return self.ocr_engine.process_document(file_bytes, filename, target_lang=target_lang, source_lang=source_lang)

    # Формирование структуры результата перевода
    def run_translate(self, ocr_output: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Сборка результатов перевода")
        return {
            "translated_text": ocr_output.get("full_text", ""),
            "tables": ocr_output.get("tables", []),
            "metadata": ocr_output.get("metadata", {})
        }

    # Очистка и подготовка текста к выводу пользователю
    def run_postprocess(self, translate_output: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Финальная обработка")
        text = translate_output.get("translated_text", "")
        cleaned_text = "\n".join([line for line in text.splitlines() if line.strip()])
        
        return {
            "cleaned_text": cleaned_text,
            "tables": translate_output.get("tables", []),
            "validation_log": ["Документ успешно обработан"],
            "metadata": translate_output.get("metadata", {}),
            "export_path": None
        }