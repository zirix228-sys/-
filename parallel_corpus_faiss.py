import os
import json
import faiss
import pickle
import numpy as np
import shutil
from os import path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional

# Класс для построения, сохранения и поиска по параллельному корпусу RU/EN
class ParallelCorpusFAISS:
    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        embedding_dim: int = 384,
        index_type: str = "IVF",
        nlist: int = 2000,
    ):
        print(f"Загружаем модель: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = embedding_dim
        self.model_name = model_name
        
        self.metadata: List[Dict] = []
        self.index_type = index_type
        self.nlist = nlist
        self.index = None

    # Создание FAISS-индекса для точного или кластерного поиска
    def _create_index(self) -> faiss.Index:
        """Создание индекса с метрикой Inner Product (для косинусного сходства)"""
        if self.index_type == "Flat":
            return faiss.IndexFlatIP(self.embedding_dim)
        
        elif self.index_type == "IVF":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            return faiss.IndexIVFFlat(quantizer, self.embedding_dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
        
        raise ValueError(f"Неподдерживаемый тип индекса: {self.index_type}")

    # Кодирование параллельного корпуса и построение векторного индекса
    def build_index(self, ru_texts: List[str], en_texts: List[str], batch_size: int = 512, save_path: str = None):
        assert len(ru_texts) == len(en_texts), "Размеры корпусов не совпадают!"
        n_samples = len(ru_texts)
        
        # 1. Генерация эмбеддингов и нормализация для косинусного сходства
        all_embeddings = []
        print(f"Кодируем {n_samples:,} пар предложений...")
        
        for i in tqdm(range(0, n_samples, batch_size), desc="Embedding"):
            ru_batch = ru_texts[i : i + batch_size]
            en_batch = en_texts[i : i + batch_size]
            
            ru_emb = self.model.encode(ru_batch, convert_to_numpy=True, show_progress_bar=False)
            en_emb = self.model.encode(en_batch, convert_to_numpy=True, show_progress_bar=False)
            
            # Объедение эмбеддингов (RU и EN идут в одну базу)
            combined = np.vstack([ru_emb, en_emb]).astype('float32')
            
            # Нормализация для косинусного сходства
            faiss.normalize_L2(combined)
            all_embeddings.append(combined)
            
            # Сохранение метаданных
            for j, (ru, en) in enumerate(zip(ru_batch, en_batch)):
                idx = i + j
                self.metadata.append({"id": idx, "ru": ru, "en": en, "lang": "ru"})
                self.metadata.append({"id": idx, "ru": ru, "en": en, "lang": "en"})

        full_matrix = np.vstack(all_embeddings)
        
        # 2. Создание и обучение индекса
        self.index = self._create_index()
        
        if self.index_type == "IVF":
            print(f"Обучаем IVF кластеры на {min(len(full_matrix), 100000)} векторах...")
            train_idx = np.random.choice(len(full_matrix), min(len(full_matrix), 100000), replace=False)
            self.index.train(full_matrix[train_idx])
            
        print("Добавляем векторы в индекс...")
        self.index.add(full_matrix)
        
        if save_path:
            self.save(save_path)

    # Поиск ближайших переводческих пар для RAG-контекста
    def search_parallel(self, query: str, k: int = 5) -> List[Dict]:
        query_emb = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_emb)
        
        distances, indices = self.index.search(query_emb, k * 3)
        
        paired_results = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1: continue
            
            meta = self.metadata[idx]
            pair_id = meta["id"]
            
            if pair_id not in paired_results:
                paired_results[pair_id] = {
                    "ru_text": meta["ru"],
                    "en_text": meta["en"],
                    "score": float(dist)
                }
            if len(paired_results) >= k: break
                
        return sorted(paired_results.values(), key=lambda x: x["score"], reverse=True)

    # Сохранение индекса, метаданных и параметров модели
    def save(self, path: str):

        directory = os.path.dirname(path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        temp_index = "parallel_corpus.index"

        faiss.write_index(self.index, temp_index)

        import shutil

        shutil.move(
            temp_index,
            f"{path}.index"
        )

        with open(f"{path}.metadata.pkl", "wb") as f:
            pickle.dump(self.metadata, f)

        config = {
            "model_name": self.model_name,
            "dim": self.embedding_dim,
            "nlist": self.nlist,
            "type": self.index_type
        }

        with open(f"{path}.config.json", "w", encoding="utf-8") as f:
            json.dump(config, f)

        print(f"База сохранена: {path}")

    @classmethod

    # Загрузка ранее построенного FAISS-индекса
    def load(cls, path: str):
        with open(f"{path}.config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
            
        instance = cls(model_name=cfg["model_name"], embedding_dim=cfg["dim"], index_type=cfg["type"], nlist=cfg["nlist"])
        instance.index = faiss.read_index(f"{path}.index")
        
        with open(f"{path}.metadata.pkl", "rb") as f:
            instance.metadata = pickle.load(f)
            
        return instance