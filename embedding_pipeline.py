#!/usr/bin/env python3
"""ChromaDB Embedding Pipeline for NASA Space Mission Data - Text Files Only"""

__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import argparse
import time
import hashlib

import chromadb
from chromadb.config import Settings
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chroma_embedding_text_only.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ChromaEmbeddingPipelineTextOnly:
    """Pipeline for creating ChromaDB collections with OpenAI embeddings - text files only."""

    def __init__(
        self,
        openai_api_key: str,
        chroma_persist_directory: str = "./chroma_db",
        collection_name: str = "nasa_space_missions_text",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.openai_api_key = openai_api_key
        self.chroma_persist_directory = chroma_persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.chunk_size = int(chunk_size)
        self.chunk_overlap = int(chunk_overlap)

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://openai.vocareum.com/v1"),
        )

        self.chroma_client = chromadb.PersistentClient(
            path=chroma_persist_directory,
            settings=Settings(allow_reset=False),
        )
        self.collection = self.chroma_client.get_or_create_collection(name=collection_name)

    def normalize_mission(self, value: str) -> str:
        value = (value or "unknown").lower()
        if value in {"apollo 11", "apollo11", "apollo_11"}:
            return "apollo_11"
        if value in {"apollo 13", "apollo13", "apollo_13"}:
            return "apollo_13"
        if "challenger" in value:
            return "challenger"
        return "unknown"

    def extract_mission_from_path(self, file_path: Path) -> str:
        s = str(file_path).lower()
        name = file_path.name.lower()

        apollo11_strong = [
            "apollo11", "apollo_11", "apollo 11",
            "a11transcript", "a11transscript",
            "as11", "as11_cm", "as11_pao", "as11_tec",
            "19900066485", "19710015566",
            "nasa_ntrs_archive_19710015566",
            "a11final-fltpln", "a11final fltpln",
            "flight_plan_hsk", "flight plan hsk",
            "apollo11transcript", "apollo11transcript",
        ]

        apollo13_strong = [
            "apollo13", "apollo_13", "apollo 13",
            "as13", "as13_cm", "as13_pao", "as13_tec"
        ]

        challenger_strong = [
            "challenger", "sts-51l", "sts_51l"
        ]

        support_words = ["transcript", "transcripts", "audio", "voice", "full_text", "full text"]

        if any(p in s or p in name for p in apollo11_strong):
            return "apollo_11"

        if any(p in s or p in name for p in apollo13_strong):
            return "apollo_13"

        if any(p in s or p in name for p in challenger_strong):
            return "challenger"

        if any(w in s or w in name for w in support_words):
            if "a11" in s or "apollo11" in s or "apollo_11" in s or "as11" in s or "19710015566" in s or "19900066485" in s:
                return "apollo_11"
            if "as13" in s or "apollo13" in s or "apollo_13" in s:
                return "apollo_13"
            if "sts-51l" in s or "sts_51l" in s or "challenger" in s:
                return "challenger"

        return "unknown"

    def extract_data_type_from_path(self, file_path: Path) -> str:
        path_str = str(file_path).lower()
        if "transcript" in path_str:
            return "transcript"
        if "textract" in path_str:
            return "textract_extracted"
        if "audio" in path_str:
            return "audio_transcript"
        if "flight_plan" in path_str:
            return "flight_plan"
        return "document"

    def extract_document_category_from_filename(self, filename: str) -> str:
        filename_lower = filename.lower()
        if "pao" in filename_lower:
            return "public_affairs_officer"
        if "cm" in filename_lower:
            return "command_module"
        if "tec" in filename_lower:
            return "technical"
        if "flight_plan" in filename_lower:
            return "flight_plan"
        if "mission_audio" in filename_lower:
            return "mission_audio"
        if "ntrs" in filename_lower:
            return "nasa_archive"
        if "19900066485" in filename_lower:
            return "technical_report"
        if "19710015566" in filename_lower:
            return "mission_report"
        if "full_text" in filename_lower:
            return "complete_document"
        return "general_document"

    def generate_chunk_id(self, file_path: Path, chunk_index: int) -> str:
        mission = self.extract_mission_from_path(file_path)
        source = file_path.stem
        raw = f"{mission}|{source}|{chunk_index}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]

    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
        if not text or not text.strip():
            return []

        text = text.strip()
        chunks = []
        start = 0
        idx = 0
        file_path = Path(metadata.get("file_path", "unknown"))

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                cut = text.rfind(". ", start, end)
                nl = text.rfind("\n", start, end)
                boundary = max(cut, nl)
                if boundary > start + max(50, self.chunk_size // 2):
                    end = boundary + 1

            chunk = text[start:end].strip()
            if chunk:
                if len(chunk) > self.chunk_size:
                    chunk = chunk[: self.chunk_size].rstrip()

                md = dict(metadata)
                md["chunk_index"] = idx
                md["chunk_id"] = self.generate_chunk_id(file_path, idx)
                chunks.append((chunk, md))
                idx += 1

            if end >= len(text):
                break

            start = max(0, end - self.chunk_overlap)

        return chunks

    def check_document_exists(self, doc_id: str) -> bool:
        try:
            result = self.collection.get(ids=[doc_id])
            return bool(result.get("ids"))
        except Exception:
            return False

    def delete_documents_by_source(self, source_pattern: str) -> int:
        try:
            all_docs = self.collection.get()
            ids_to_delete = []
            for i, metadata in enumerate(all_docs.get("metadatas", [])):
                if source_pattern in metadata.get("source", "") or source_pattern in metadata.get("file_path", ""):
                    ids_to_delete.append(all_docs["ids"][i])
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                return len(ids_to_delete)
            return 0
        except Exception as e:
            logger.error(f"Error deleting documents by source: {e}")
            return 0

    def get_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.embedding_model, input=texts)
        return [list(item.embedding) for item in resp.data]

    def process_text_file(self, file_path: Path) -> List[Tuple[str, Dict[str, Any]]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                return []

            metadata = {
                "source": file_path.stem,
                "file_path": str(file_path),
                "file_type": "text",
                "content_type": "full_text",
                "mission": self.extract_mission_from_path(file_path),
                "data_type": self.extract_data_type_from_path(file_path),
                "document_category": self.extract_document_category_from_filename(file_path.name),
                "file_size": len(content),
                "processed_timestamp": datetime.now().isoformat(),
            }
            return self.chunk_text(content, metadata)
        except Exception as e:
            logger.error(f"Error processing text file {file_path}: {e}")
            return []

    def scan_text_files_only(self, base_path: str) -> List[Path]:
        base_path = Path(base_path)
        files_to_process = []
        data_dirs = ["apollo11", "apollo13", "challenger"]

        for data_dir in data_dirs:
            dir_path = base_path / data_dir
            if dir_path.exists():
                logger.info(f"Scanning directory: {dir_path}")
                text_files = sorted(dir_path.glob("**/*.txt"))
                text_files = [
                    fp for fp in text_files
                    if not fp.name.startswith(".")
                    and "summary" not in fp.name.lower()
                    and fp.suffix.lower() == ".txt"
                ]
                files_to_process.extend(text_files)
                logger.info(f"Found {len(text_files)} text files in {data_dir}")

        logger.info(f"Total text files to process: {len(files_to_process)}")
        return files_to_process

    def add_documents_to_collection(
        self,
        documents: List[Tuple[str, Dict[str, Any]]],
        file_path: Path,
        batch_size: int = 50,
        update_mode: str = "skip",
    ) -> Dict[str, int]:
        if not documents:
            return {"added": 0, "updated": 0, "skipped": 0}

        stats = {"added": 0, "updated": 0, "skipped": 0}

        if update_mode == "replace":
            deleted = self.delete_documents_by_source(file_path.stem)
            logger.info(f"Deleted {deleted} existing docs for source {file_path.stem}")

        texts_to_add = []
        metas_to_add = []
        ids_to_add = []

        for text, metadata in documents:
            doc_id = metadata.get("chunk_id") or self.generate_chunk_id(file_path, int(metadata.get("chunk_index", 0)))
            exists = self.check_document_exists(doc_id)

            if exists and update_mode == "skip":
                stats["skipped"] += 1
                continue

            md = dict(metadata)
            md["chunk_id"] = doc_id

            ids_to_add.append(doc_id)
            texts_to_add.append(text)
            metas_to_add.append(md)

            if exists and update_mode == "update":
                stats["updated"] += 1
            else:
                stats["added"] += 1

        for start in range(0, len(texts_to_add), batch_size):
            batch_texts = texts_to_add[start:start + batch_size]
            batch_ids = ids_to_add[start:start + batch_size]
            batch_metas = metas_to_add[start:start + batch_size]
            batch_embeds = self.get_embedding_batch(batch_texts)

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                metadatas=batch_metas,
                embeddings=batch_embeds,
            )

        return stats

    def process_all_text_data(self, base_path: str, update_mode: str = "skip") -> Dict[str, Any]:
        stats = {
            "files_processed": 0,
            "documents_added": 0,
            "documents_updated": 0,
            "documents_skipped": 0,
            "errors": 0,
            "total_chunks": 0,
            "missions": {},
        }

        files = self.scan_text_files_only(base_path)

        for file_path in files:
            try:
                docs = self.process_text_file(file_path)
                file_stats = self.add_documents_to_collection(docs, file_path, update_mode=update_mode)

                mission = self.extract_mission_from_path(file_path)
                stats["missions"].setdefault(
                    mission,
                    {"files": 0, "chunks": 0, "added": 0, "updated": 0, "skipped": 0},
                )

                stats["missions"][mission]["files"] += 1
                stats["missions"][mission]["chunks"] += len(docs)
                stats["missions"][mission]["added"] += file_stats["added"]
                stats["missions"][mission]["updated"] += file_stats["updated"]
                stats["missions"][mission]["skipped"] += file_stats["skipped"]

                stats["files_processed"] += 1
                stats["documents_added"] += file_stats["added"]
                stats["documents_updated"] += file_stats["updated"]
                stats["documents_skipped"] += file_stats["skipped"]
                stats["total_chunks"] += len(docs)

                logger.info(
                    f"Processed {file_path.name}: chunks={len(docs)}, "
                    f"added={file_stats['added']}, updated={file_stats['updated']}, skipped={file_stats['skipped']}"
                )
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                stats["errors"] += 1

        return stats

    def get_collection_info(self) -> Dict[str, Any]:
        try:
            return {
                "collection_name": self.collection_name,
                "document_count": self.collection.count(),
                "persist_directory": self.chroma_persist_directory,
                "embedding_model": self.embedding_model,
            }
        except Exception as e:
            return {"error": str(e)}

    def query_collection(self, query_text: str, n_results: int = 5) -> Dict[str, Any]:
        try:
            return self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                include=["documents", "metadatas", "distances", "ids"],
            )
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            return {"error": str(e)}

    def get_collection_stats(self) -> Dict[str, Any]:
        try:
            all_docs = self.collection.get()
            if not all_docs["metadatas"]:
                return {"error": "No documents in collection"}

            stats = {
                "total_documents": len(all_docs["metadatas"]),
                "missions": {},
                "data_types": {},
                "document_categories": {},
                "file_types": {},
            }

            for metadata in all_docs["metadatas"]:
                mission = metadata.get("mission", "unknown")
                data_type = metadata.get("data_type", "unknown")
                doc_category = metadata.get("document_category", "unknown")
                file_type = metadata.get("file_type", "unknown")

                stats["missions"][mission] = stats["missions"].get(mission, 0) + 1
                stats["data_types"][data_type] = stats["data_types"].get(data_type, 0) + 1
                stats["document_categories"][doc_category] = stats["document_categories"].get(doc_category, 0) + 1
                stats["file_types"][file_type] = stats["file_types"].get(file_type, 0) + 1

            return stats
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="ChromaDB Embedding Pipeline for NASA Data")
    parser.add_argument("--data-path", default=".", help="Path to data directories")
    parser.add_argument("--openai-key", required=True, help="OpenAI API key")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="ChromaDB persist directory")
    parser.add_argument("--collection-name", default="nasa_space_missions_text", help="Collection name")
    parser.add_argument("--embedding-model", default="text-embedding-3-small", help="OpenAI embedding model")
    parser.add_argument("--chunk-size", type=int, default=500, help="Text chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap size")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for processing")
    parser.add_argument("--update-mode", choices=["skip", "update", "replace"], default="skip",
                        help="How to handle existing documents: skip, update, or replace")
    parser.add_argument("--test-query", help="Test query after processing")
    parser.add_argument("--stats-only", action="store_true", help="Only show collection statistics")
    parser.add_argument("--delete-source", help="Delete all documents from a specific source pattern")
    args = parser.parse_args()

    logger.info("Initializing ChromaDB Embedding Pipeline...")
    pipeline = ChromaEmbeddingPipelineTextOnly(
        args.openai_key,
        args.chroma_dir,
        args.collection_name,
        args.embedding_model,
        args.chunk_size,
        args.chunk_overlap,
    )

    if args.delete_source:
        deleted = pipeline.delete_documents_by_source(args.delete_source)
        print(deleted)
        return

    if args.stats_only:
        stats = pipeline.get_collection_stats()
        logger.info("Collection Statistics:")
        for key, value in stats.items():
            logger.info(f"{key}: {value}")
        return

    start_time = time.time()
    stats = pipeline.process_all_text_data(args.data_path, update_mode=args.update_mode)
    processing_time = time.time() - start_time

    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Files processed: {stats['files_processed']}")
    logger.info(f"Total chunks created: {stats['total_chunks']}")
    logger.info(f"Documents added to collection: {stats['documents_added']}")
    logger.info(f"Documents updated in collection: {stats['documents_updated']}")
    logger.info(f"Documents skipped (already exist): {stats['documents_skipped']}")
    logger.info(f"Errors: {stats['errors']}")
    logger.info(f"Processing time: {processing_time:.2f} seconds")
    logger.info("\nMission breakdown:")

    for mission, mission_stats in stats["missions"].items():
        logger.info(f"  {mission}: {mission_stats['files']} files, {mission_stats['chunks']} chunks")
        logger.info(
            f"    Added: {mission_stats['added']}, Updated: {mission_stats['updated']}, Skipped: {mission_stats['skipped']}"
        )

    collection_info = pipeline.get_collection_info()
    logger.info(f"\nCollection: {collection_info.get('collection_name', 'N/A')}")
    logger.info(f"Total documents in collection: {collection_info.get('document_count', 'N/A')}")

    if args.test_query:
        results = pipeline.query_collection(args.test_query)
        if results and "documents" in results:
            logger.info(f"Found {len(results['documents'][0])} results:")
            for i, doc in enumerate(results["documents"][0][:3]):
                logger.info(f"Result {i+1}: {doc[:200]}...")


if __name__ == "__main__":
    main()