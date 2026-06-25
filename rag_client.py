__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import chromadb
from chromadb.config import Settings
from typing import Dict, List, Optional
from pathlib import Path


def _safe_collection_count(collection) -> int:
    try:
        return int(collection.count())
    except Exception:
        try:
            return len(collection.get().get('ids', []))
        except Exception:
            return 0


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path('.')
    dirs = [p for p in current_dir.iterdir() if p.is_dir() and p.name.startswith('chroma')]
    if not dirs:
        dirs = [p for p in current_dir.iterdir() if p.is_dir()]

    for dir_path in dirs:
        try:
            client = chromadb.PersistentClient(path=str(dir_path))
            collections = client.list_collections()
            if not collections:
                continue
            for col in collections:
                col_name = getattr(col, 'name', str(col))
                try:
                    collection = client.get_collection(name=col_name)
                    count = _safe_collection_count(collection)
                except Exception:
                    count = 0
                key = f"{dir_path.name}::{col_name}"
                backends[key] = {
                    'directory': str(dir_path),
                    'collection_name': col_name,
                    'display_name': f'{dir_path.name} / {col_name} ({count} docs)',
                    'document_count': str(count),
                }
        except Exception as e:
            key = f'{dir_path.name}::inaccessible'
            backends[key] = {
                'directory': str(dir_path),
                'collection_name': '',
                'display_name': f'{dir_path.name} (unavailable: {str(e)[:50]})',
                'document_count': '0',
            }
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""
    try:
        client = chromadb.PersistentClient(path=chroma_dir, settings=Settings(allow_reset=False))
        collection = client.get_or_create_collection(name=collection_name)
        return collection, True, ''
    except Exception as e:
        return None, False, str(e)


def retrieve_documents(collection, query: str, n_results: int = 3,
                      mission_filter: Optional[str] = None) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""
    where = None
    if mission_filter and str(mission_filter).lower() not in {'all', 'any', 'none', ''}:
        where = {'mission': mission_filter}
    try:
        return collection.query(
            query_texts=[query],
            n_results=int(n_results),
            where=where,
            include=['documents', 'metadatas', 'distances', 'ids']
        )
    except Exception:
        return None


def format_context(documents: List[str], metadatas: List[Dict]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ''
    seen = set()
    context_parts = ['Retrieved NASA mission context:']
    items = []
    for doc, meta in zip(documents, metadatas):
        text = str(doc).strip()
        if not text:
            continue
        fingerprint = text[:200].lower()
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        score = meta.get('_distance', meta.get('distance', ''))
        items.append((score, text, meta))

    def score_key(x):
        s = x[0]
        try:
            return float(s)
        except Exception:
            return 0.0

    items.sort(key=score_key)
    for idx, (_, doc, meta) in enumerate(items, start=1):
        mission = str(meta.get('mission', 'unknown')).replace('_', ' ').title()
        source = str(meta.get('source', meta.get('file_path', 'unknown')))
        category = str(meta.get('document_category', meta.get('category', 'general'))).replace('_', ' ').title()
        header = f'[{idx}] Mission: {mission} | Source: {source} | Category: {category}'
        context_parts.append(header)
        if len(doc) > 1200:
            doc = doc[:1200].rstrip() + '...'
        context_parts.append(doc)
    return '\n\n'.join(context_parts)