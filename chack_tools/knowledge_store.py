"""Curated, local research knowledge backed by Qdrant.

Qdrant is a derived retrieval index, not the evidence authority.  Source hashes
and ingest state live in a small SQLite catalogue so an embedding model or
chunking change can be rebuilt deterministically from approved source files.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from fnmatch import fnmatchcase
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .config import ToolsConfig
from .telemetry import run_with_tool_logging

try:
    from agents import function_tool
except ImportError:  # pragma: no cover - optional runtime dependency
    function_tool = None


KNOWLEDGE_MODES = {"off", "read", "write", "read_write"}
DISPOSITIONS = {"ingest_candidate", "archive_only", "discard", "rejected", "approved"}
TEXT_EXTENSIONS = {
    ".md", ".markdown", ".txt", ".json", ".jsonl", ".yaml", ".yml",
    ".csv", ".tsv", ".html", ".htm", ".xml", ".rst", ".log",
}
# Only formats whose exact decoded text is the authoritative representation may
# be removed after Qdrant payload verification. Markup can contain structure or
# attributes that are deliberately absent from retrieval text.
DELETABLE_TEXT_EXTENSIONS = TEXT_EXTENSIONS - {".html", ".htm", ".xml"}
EXTRACTABLE_EXTENSIONS = TEXT_EXTENSIONS | {".pdf"}
POLICY_PREFIX = "<!-- chack-knowledge-policy:"
POLICY_SUFFIX = " -->"


def normalize_knowledge_mode(value: Any, default: str = "off") -> str:
    mode = str(value or default or "off").strip().lower().replace("-", "_")
    if mode not in KNOWLEDGE_MODES:
        raise ValueError(f"knowledge_mode must be one of {sorted(KNOWLEDGE_MODES)}")
    return mode


def normalize_knowledge_base(value: Any) -> str:
    base = re.sub(r"[^a-z0-9._-]+", "-", str(value or "").strip().lower()).strip("-._")
    if value and not base:
        raise ValueError("knowledge_base contains no usable characters")
    return base


def knowledge_can_read(mode: str) -> bool:
    return normalize_knowledge_mode(mode) in {"read", "read_write"}


def knowledge_can_write(mode: str) -> bool:
    return normalize_knowledge_mode(mode) in {"write", "read_write"}


def encode_knowledge_policy(prompt: str, *, mode: str, knowledge_base: str) -> str:
    payload = json.dumps(
        {"mode": normalize_knowledge_mode(mode), "knowledge_base": normalize_knowledge_base(knowledge_base)},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return f"{POLICY_PREFIX}{payload}{POLICY_SUFFIX}\n{str(prompt or '').lstrip()}"


def decode_knowledge_policy(prompt: str) -> tuple[dict[str, str], str]:
    text = str(prompt or "")
    first, separator, rest = text.partition("\n")
    if not (first.startswith(POLICY_PREFIX) and first.endswith(POLICY_SUFFIX)):
        return {"mode": "off", "knowledge_base": ""}, text
    raw = first[len(POLICY_PREFIX):-len(POLICY_SUFFIX)]
    try:
        payload = json.loads(raw)
        mode = normalize_knowledge_mode(payload.get("mode"))
        base = normalize_knowledge_base(payload.get("knowledge_base"))
    except (ValueError, TypeError, json.JSONDecodeError):
        return {"mode": "off", "knowledge_base": ""}, text
    return {"mode": mode, "knowledge_base": base}, rest if separator else ""


def knowledge_policy_instruction(mode: str, knowledge_base: str) -> str:
    mode = normalize_knowledge_mode(mode)
    base = normalize_knowledge_base(knowledge_base)
    if mode == "off" or not base:
        return ""
    write_line = (
        "Researchers must classify every retained key_artifact exactly once in knowledge_candidates. "
        "The administrator must review every candidate, "
        "return knowledge_decisions using paths relative to its evidence root, and approve only durable, sourced, "
        "non-duplicative evidence. Runtime code, not any agent, performs ingestion and deletion."
        if knowledge_can_write(mode)
        else "Do not nominate or write new knowledge for this ephemeral/read-only run."
    )
    read_line = (
        f"Search the read-only knowledge base '{base}' before repeating prior work and cite returned source paths. "
        "Treat retrieved passages as leads: reopen primary evidence for central claims."
        if knowledge_can_read(mode)
        else "This run must not read prior knowledge."
    )
    return f"\n\n### KNOWLEDGE POLICY\nMode: {mode}; base: {base}.\n{read_line}\n{write_line}\n"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "svg", "noscript"}:
            self._ignored += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "svg", "noscript"} and self._ignored:
            self._ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored and data.strip():
            self.parts.append(data.strip())


def extract_document_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        return "\n\n".join((page.extract_text() or "").strip() for page in PdfReader(str(path)).pages).strip()
    raw = path.read_text(encoding="utf-8", errors="replace")
    if suffix in {".html", ".htm", ".xml"}:
        parser = _TextExtractor()
        parser.feed(raw)
        return "\n".join(parser.parts)
    return raw


def chunk_text(text: str, *, size: int, overlap: int) -> list[str]:
    normalized = re.sub(r"[ \t]+", " ", str(text or "")).replace("\r\n", "\n").strip()
    if not normalized:
        return []
    size = max(300, int(size or 1800))
    overlap = max(0, min(int(overlap or 0), size // 2))
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + size)
        if end < len(normalized):
            split = max(normalized.rfind("\n\n", start + size // 2, end), normalized.rfind(". ", start + size // 2, end))
            if split > start:
                end = split + (2 if normalized[split:split + 2] == ". " else 0)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


@dataclass(frozen=True)
class KnowledgeDocument:
    path: Path
    source_path: str
    title: str = ""
    source_url: str = ""
    evidence_type: str = ""
    source_group: str = ""
    disposition: str = "approved"
    delete_after_ingest: bool = False
    delete_extracted_after_ingest: bool = False


def _should_delete_ingested_artifact(document: KnowledgeDocument, path: Path) -> bool:
    suffix = path.suffix.lower()
    return (
        (document.delete_after_ingest and suffix in DELETABLE_TEXT_EXTENSIONS)
        or (document.delete_extracted_after_ingest and suffix in EXTRACTABLE_EXTENSIONS)
    )


def _remove_artifact_manifest_entries(path: Path, *, stop: Path | None = None) -> None:
    """Remove exact stale rows from any owned artifact manifests above a file."""

    try:
        from .research_artifacts import remove_research_artifact_manifest_entry
    except ImportError:
        return
    for parent in path.parents:
        if stop is not None:
            try:
                parent.relative_to(stop)
            except ValueError:
                break
        try:
            rel = path.relative_to(parent)
        except ValueError:
            continue
        remove_research_artifact_manifest_entry(parent, rel)
        if stop is not None and parent == stop:
            break


class KnowledgeStore:
    def __init__(self, config: ToolsConfig):
        self.config = config
        self.url = str(config.knowledge_qdrant_url or "http://127.0.0.1:6333").rstrip("/")
        self.collection = str(config.knowledge_collection or "chack_research_knowledge")
        self.model_name = str(config.knowledge_embedding_model)
        self._client: Any = None
        self._dense: Any = None
        self._sparse: Any = None
        self._collection_ready = False

    def _embedding_runtime_signature(self) -> str:
        """Return the runtime component that can change vector semantics.

        The configured model name alone is insufficient: FastEmbed has changed
        pooling behaviour for existing model names between releases.  Persisting
        the package version lets status/reindex detect mixed vector generations.
        """

        try:
            fastembed_version = importlib.metadata.version("fastembed")
        except importlib.metadata.PackageNotFoundError:  # pragma: no cover - model load fails first
            fastembed_version = "missing"
        return f"fastembed={fastembed_version}"

    def _allowed_base(self, knowledge_base: str) -> str:
        base = normalize_knowledge_base(knowledge_base or self.config.knowledge_base)
        allowed = {normalize_knowledge_base(item) for item in (self.config.knowledge_allowed_bases or []) if item}
        if not base:
            raise ValueError("knowledge_base is required")
        if allowed and base not in allowed:
            raise ValueError(f"knowledge_base '{base}' is not allowed")
        return base

    def _qdrant(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(url=self.url, timeout=60)
        return self._client

    def _models(self) -> tuple[Any, Any]:
        if self._dense is None or self._sparse is None:
            from fastembed import SparseTextEmbedding, TextEmbedding

            self._dense = TextEmbedding(model_name=self.model_name)
            self._sparse = SparseTextEmbedding(model_name="Qdrant/bm25")
        return self._dense, self._sparse

    def _ensure_collection(self) -> None:
        from qdrant_client import models

        if self._collection_ready:
            return
        client = self._qdrant()
        if not client.collection_exists(self.collection):
            supported = {row["model"]: int(row["dim"]) for row in self._models()[0].list_supported_models()}
            dimension = supported.get(self.model_name)
            if not dimension:
                probe = next(self._dense.embed(["dimension probe"]))
                dimension = len(probe)
            client.create_collection(
                collection_name=self.collection,
                vectors_config={"dense": models.VectorParams(size=dimension, distance=models.Distance.COSINE)},
                sparse_vectors_config={"lexical": models.SparseVectorParams(modifier=models.Modifier.IDF)},
            )
        for field in (
            "knowledge_base", "source_id", "source_path", "content_hash", "evidence_type",
            "source_group", "document_kind", "section", "research_slug",
        ):
            client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
        self._collection_ready = True

    def _catalog(self) -> sqlite3.Connection:
        path = Path(str(self.config.knowledge_catalog_path)).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=30)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """CREATE TABLE IF NOT EXISTS sources (
                knowledge_base TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_url TEXT NOT NULL,
                title TEXT NOT NULL,
                evidence_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                byte_size INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL,
                embedding_model TEXT NOT NULL,
                embedding_runtime TEXT NOT NULL DEFAULT '',
                chunk_size INTEGER NOT NULL,
                chunk_overlap INTEGER NOT NULL,
                ingested_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                source_deleted INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (knowledge_base, source_id)
            )"""
        )
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(sources)")}
        if "embedding_runtime" not in columns:
            connection.execute(
                "ALTER TABLE sources ADD COLUMN embedding_runtime TEXT NOT NULL DEFAULT ''"
            )
        return connection

    def ingest_documents(self, knowledge_base: str, documents: Iterable[KnowledgeDocument]) -> dict[str, Any]:
        from qdrant_client import models

        base = self._allowed_base(knowledge_base)
        self._ensure_collection()
        dense_model, sparse_model = self._models()
        embedding_runtime = self._embedding_runtime_signature()
        client = self._qdrant()
        indexed: list[dict[str, Any]] = []
        skipped: list[dict[str, str]] = []
        deleted: list[str] = []
        with self._catalog() as catalog:
            for document in documents:
                path = document.path.expanduser().resolve()
                if document.disposition != "approved":
                    skipped.append({"path": str(path), "reason": f"disposition={document.disposition}"})
                    continue
                if not path.is_file():
                    skipped.append({"path": str(path), "reason": "file_not_found"})
                    continue
                text = extract_document_text(path)
                if not text.strip():
                    skipped.append({"path": str(path), "reason": "no_extractable_text"})
                    continue
                content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                source_id = hashlib.sha256(f"{base}\0{document.source_path}".encode()).hexdigest()
                chunks = chunk_text(
                    text,
                    size=self.config.knowledge_chunk_size_chars,
                    overlap=self.config.knowledge_chunk_overlap_chars,
                )
                current = catalog.execute(
                    "SELECT content_hash, embedding_model, embedding_runtime, chunk_size, chunk_overlap, chunk_count FROM sources WHERE knowledge_base=? AND source_id=?",
                    (base, source_id),
                ).fetchone()
                signature = (
                    content_hash,
                    self.model_name,
                    embedding_runtime,
                    int(self.config.knowledge_chunk_size_chars),
                    int(self.config.knowledge_chunk_overlap_chars),
                )
                if current and tuple(current[:5]) == signature:
                    skipped.append({"path": str(path), "reason": "unchanged"})
                    if (
                        _should_delete_ingested_artifact(document, path)
                        and self._verify_source_payload(base, source_id, content_hash, int(current[5]))
                    ):
                        path.unlink()
                        _remove_artifact_manifest_entries(path)
                        catalog.execute(
                            "UPDATE sources SET source_deleted=1 WHERE knowledge_base=? AND source_id=?",
                            (base, source_id),
                        )
                        deleted.append(str(path))
                    catalog.commit()
                    continue
                selector = models.Filter(
                    must=[
                        models.FieldCondition(key="knowledge_base", match=models.MatchValue(value=base)),
                        models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id)),
                    ]
                )
                if current:
                    client.delete(self.collection, selector, wait=True)
                dense_vectors = list(dense_model.embed(chunks))
                sparse_vectors = list(sparse_model.embed(chunks))
                points = []
                path_parts = Path(document.source_path).parts
                is_research = bool(path_parts and path_parts[0] == "researches")
                document_kind = "research_evidence" if is_research else "published_page"
                section = path_parts[0] if path_parts else ""
                research_slug = path_parts[1] if is_research and len(path_parts) > 1 else ""
                for index, (chunk, dense, sparse) in enumerate(zip(chunks, dense_vectors, sparse_vectors)):
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"chack:{base}:{source_id}:{content_hash}:{index}"))
                    points.append(
                        models.PointStruct(
                            id=point_id,
                            vector={
                                "dense": dense.tolist(),
                                "lexical": models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist()),
                            },
                            payload={
                                "knowledge_base": base,
                                "source_id": source_id,
                                "source_path": document.source_path,
                                "source_url": document.source_url,
                                "title": document.title or path.stem,
                                "evidence_type": document.evidence_type,
                                "source_group": document.source_group,
                                "document_kind": document_kind,
                                "section": section,
                                "research_slug": research_slug,
                                "content_hash": content_hash,
                                "chunk_index": index,
                                "chunk_count": len(chunks),
                                "text": chunk,
                                # One copy of the extracted source makes deletion of
                                # explicitly disposable text reversible and verifiable.
                                "source_text": text if index == 0 else "",
                            },
                        )
                    )
                for offset in range(0, len(points), 64):
                    client.upsert(self.collection, points[offset:offset + 64], wait=True)
                catalog.execute(
                    """INSERT INTO sources (
                        knowledge_base,source_id,source_path,source_url,title,evidence_type,content_hash,
                        byte_size,chunk_count,embedding_model,chunk_size,chunk_overlap,source_deleted
                        ,embedding_runtime
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?)
                    ON CONFLICT(knowledge_base,source_id) DO UPDATE SET
                        source_path=excluded.source_path,source_url=excluded.source_url,title=excluded.title,
                        evidence_type=excluded.evidence_type,content_hash=excluded.content_hash,
                        byte_size=excluded.byte_size,chunk_count=excluded.chunk_count,
                        embedding_model=excluded.embedding_model,chunk_size=excluded.chunk_size,
                        chunk_overlap=excluded.chunk_overlap,embedding_runtime=excluded.embedding_runtime,
                        ingested_at=CURRENT_TIMESTAMP,source_deleted=0""",
                    (
                        base, source_id, document.source_path, document.source_url, document.title or path.stem,
                        document.evidence_type, content_hash, path.stat().st_size, len(chunks), self.model_name,
                        int(self.config.knowledge_chunk_size_chars), int(self.config.knowledge_chunk_overlap_chars),
                        embedding_runtime,
                    ),
                )
                indexed.append({"path": str(path), "source_id": source_id, "chunks": len(chunks), "sha256": content_hash})
                if _should_delete_ingested_artifact(document, path):
                    # The default path only removes exact plain text. A profile
                    # may explicitly accept loss of original markup/layout for
                    # disposable HTML/XML/PDF queue downloads.
                    if self._verify_source_payload(base, source_id, content_hash, len(chunks)):
                        path.unlink()
                        _remove_artifact_manifest_entries(path)
                        catalog.execute(
                            "UPDATE sources SET source_deleted=1 WHERE knowledge_base=? AND source_id=?",
                            (base, source_id),
                        )
                        deleted.append(str(path))
                # Qdrant upserts are already durable per source. Commit the
                # matching catalog row per source so interrupted seed runs resume cleanly.
                catalog.commit()
            catalog.commit()
        return {"knowledge_base": base, "indexed": indexed, "skipped": skipped, "deleted": deleted}

    def _verify_source_payload(self, base: str, source_id: str, content_hash: str, chunk_count: int) -> bool:
        from qdrant_client import models

        records, _ = self._qdrant().scroll(
            self.collection,
            scroll_filter=models.Filter(must=[
                models.FieldCondition(key="knowledge_base", match=models.MatchValue(value=base)),
                models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id)),
                models.FieldCondition(key="content_hash", match=models.MatchValue(value=content_hash)),
            ]),
            limit=max(1, chunk_count + 1),
            with_payload=["chunk_index", "text", "source_text"],
            with_vectors=False,
        )
        exact_sources = [
            str((record.payload or {}).get("source_text") or "")
            for record in records
            if str((record.payload or {}).get("source_text") or "")
        ]
        return (
            len(records) == chunk_count
            and all(str((record.payload or {}).get("text") or "") for record in records)
            and len(exact_sources) == 1
            and hashlib.sha256(exact_sources[0].encode("utf-8")).hexdigest() == content_hash
        )

    def search(
        self,
        query: str,
        knowledge_base: str,
        *,
        vector_limit: int = 0,
        exact_limit: int = 0,
        max_chars: int = 0,
    ) -> dict[str, Any]:
        from qdrant_client import models

        base = self._allowed_base(knowledge_base)
        query = str(query or "").strip()
        if not query:
            raise ValueError("query cannot be empty")
        self._ensure_collection()
        dense_model, sparse_model = self._models()
        dense = next(dense_model.query_embed(query)).tolist()
        sparse = next(sparse_model.query_embed(query))
        query_filter = models.Filter(
            must=[models.FieldCondition(key="knowledge_base", match=models.MatchValue(value=base))]
        )
        # These settings are operator-owned context limits, not merely defaults.
        # Agents may request fewer results for a narrow lookup, but cannot expand
        # a search until it floods their own context window.
        configured_vector_limit = max(0, min(int(self.config.knowledge_vector_results), 50))
        configured_exact_limit = max(0, min(int(self.config.knowledge_exact_results), 50))
        vector_limit = (
            configured_vector_limit
            if int(vector_limit or 0) <= 0
            else min(int(vector_limit), configured_vector_limit)
        )
        exact_limit = (
            configured_exact_limit
            if int(exact_limit or 0) <= 0
            else min(int(exact_limit), configured_exact_limit)
        )
        payload_fields = [
            "source_path", "source_url", "title", "evidence_type", "source_group",
            "document_kind", "section", "research_slug", "chunk_index", "content_hash", "text",
        ]
        client = self._qdrant()
        vector_hits = client.query_points(
            self.collection, query=dense, using="dense", query_filter=query_filter,
            limit=vector_limit, with_payload=payload_fields,
        ).points if vector_limit else []
        exact_hits = client.query_points(
            self.collection,
            query=models.SparseVector(indices=sparse.indices.tolist(), values=sparse.values.tolist()),
            using="lexical", query_filter=query_filter, limit=exact_limit, with_payload=payload_fields,
        ).points if exact_limit else []
        merged: dict[str, dict[str, Any]] = {}
        for kind, hits in (("vector", vector_hits), ("exact", exact_hits)):
            for rank, hit in enumerate(hits, start=1):
                payload = dict(hit.payload or {})
                key = str(hit.id)
                row = merged.setdefault(key, {
                    "source_path": payload.get("source_path", ""),
                    "source_url": payload.get("source_url", ""),
                    "title": payload.get("title", ""),
                    "evidence_type": payload.get("evidence_type", ""),
                    "source_group": payload.get("source_group", ""),
                    "document_kind": payload.get("document_kind", ""),
                    "section": payload.get("section", ""),
                    "research_slug": payload.get("research_slug", ""),
                    "chunk_index": payload.get("chunk_index", 0),
                    "content_hash": payload.get("content_hash", ""),
                    "text": payload.get("text", ""),
                    "vector_rank": None,
                    "exact_rank": None,
                    "rrf_score": 0.0,
                })
                row[f"{kind}_rank"] = rank
                row["rrf_score"] += 1.0 / (60.0 + rank)
        results = sorted(merged.values(), key=lambda row: (-float(row["rrf_score"]), str(row["source_path"])))
        configured_char_budget = max(1000, int(self.config.knowledge_max_return_chars))
        char_budget = (
            configured_char_budget
            if int(max_chars or 0) <= 0
            else min(max(1000, int(max_chars)), configured_char_budget)
        )
        bounded: list[dict[str, Any]] = []
        used = 0
        for row in results:
            text = str(row.get("text") or "")
            remaining = char_budget - used
            if remaining <= 0:
                break
            if len(text) > remaining:
                text = (text[:max(0, remaining - 1)].rstrip() + "…") if remaining > 1 else "…"
            row["text"] = text
            used += len(text)
            bounded.append(row)
        return {
            "knowledge_base": base,
            "query": query,
            "vector_requested": vector_limit,
            "exact_requested": exact_limit,
            "max_returned_text_chars": char_budget,
            "results": bounded,
            "result_count": len(bounded),
        }

    def status(self, knowledge_base: str = "") -> dict[str, Any]:
        base = self._allowed_base(knowledge_base)
        self._ensure_collection()
        embedding_runtime = self._embedding_runtime_signature()
        with self._catalog() as catalog:
            row = catalog.execute(
                "SELECT COUNT(*), COALESCE(SUM(chunk_count),0), COALESCE(SUM(byte_size),0), COALESCE(SUM(source_deleted),0) FROM sources WHERE knowledge_base=?",
                (base,),
            ).fetchone() or (0, 0, 0, 0)
            compatible_sources = int((catalog.execute(
                """SELECT COUNT(*) FROM sources
                   WHERE knowledge_base=? AND embedding_model=? AND embedding_runtime=?
                     AND chunk_size=? AND chunk_overlap=?""",
                (
                    base,
                    self.model_name,
                    embedding_runtime,
                    int(self.config.knowledge_chunk_size_chars),
                    int(self.config.knowledge_chunk_overlap_chars),
                ),
            ).fetchone() or (0,))[0])
        return {
            "knowledge_base": base,
            "collection": self.collection,
            "sources": int(row[0]),
            "chunks": int(row[1]),
            "source_bytes": int(row[2]),
            "source_files_deleted_after_verified_ingest": int(row[3]),
            "embedding_model": self.model_name,
            "embedding_runtime": embedding_runtime,
            "embedding_compatible_sources": compatible_sources,
            "embedding_incompatible_sources": max(0, int(row[0]) - compatible_sources),
            "chunk_size_chars": int(self.config.knowledge_chunk_size_chars),
            "chunk_overlap_chars": int(self.config.knowledge_chunk_overlap_chars),
            "vector_results": int(self.config.knowledge_vector_results),
            "exact_results": int(self.config.knowledge_exact_results),
        }

    def documents_from_seed_manifest(self, knowledge_base: str) -> list[KnowledgeDocument]:
        base = self._allowed_base(knowledge_base)
        raw_path = str((self.config.knowledge_seed_manifests or {}).get(base) or "")
        if not raw_path:
            raise ValueError(f"no seed manifest configured for knowledge_base '{base}'")
        manifest_path = Path(raw_path).expanduser().resolve()
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        manifest_base = normalize_knowledge_base(payload.get("knowledge_base"))
        if manifest_base != base:
            raise ValueError("seed manifest knowledge_base does not match the requested base")
        root = Path(str(payload.get("root") or "")).expanduser().resolve()
        allowed_root = str((self.config.knowledge_allowed_roots or {}).get(base) or "")
        if not allowed_root or root != Path(allowed_root).expanduser().resolve():
            raise ValueError("seed manifest root is not the configured allowed root")
        global_excludes = [str(item) for item in (payload.get("exclude") or [])]
        source_groups = payload.get("sources")
        if not isinstance(source_groups, list):
            source_groups = [{
                "name": "curated-local-research",
                "evidence_type": "curated_local_research",
                "include": payload.get("include") or [],
            }]
        documents: list[KnowledgeDocument] = []
        seen: set[Path] = set()
        for raw_group in source_groups:
            if not isinstance(raw_group, Mapping):
                continue
            group = normalize_knowledge_base(raw_group.get("name") or "curated-local-research")
            evidence_type = str(raw_group.get("evidence_type") or "curated_local_research").strip()
            includes = [str(item) for item in (raw_group.get("include") or [])]
            excludes = global_excludes + [str(item) for item in (raw_group.get("exclude") or [])]
            candidates: set[Path] = set()
            for pattern in includes:
                candidates.update(path for path in root.glob(pattern) if path.is_file())
            for path in sorted(candidates):
                if path in seen:
                    continue
                rel = path.relative_to(root).as_posix()
                if any(fnmatchcase(rel.casefold(), pattern.casefold()) for pattern in excludes):
                    continue
                if path.suffix.lower() not in TEXT_EXTENSIONS and path.suffix.lower() != ".pdf":
                    continue
                seen.add(path)
                documents.append(KnowledgeDocument(
                    path=path,
                    source_path=rel,
                    evidence_type=evidence_type,
                    source_group=group,
                ))
        return documents

    def ingest_seed_manifest(self, knowledge_base: str) -> dict[str, Any]:
        documents = self.documents_from_seed_manifest(knowledge_base)
        result = self.ingest_documents(knowledge_base, documents)
        base = self._allowed_base(knowledge_base)
        desired_source_ids = {
            hashlib.sha256(f"{base}\0{document.source_path}".encode()).hexdigest()
            for document in documents
        }
        pruned: list[str] = []
        with self._catalog() as catalog:
            rows = catalog.execute(
                "SELECT source_id, source_path FROM sources WHERE knowledge_base=?",
                (base,),
            ).fetchall()
            for source_id, source_path in rows:
                # Runtime-approved sources use reserved prefixes and are not
                # governed by the repository seed manifest.
                if str(source_path).startswith(("source-url/", "research-artifact/")):
                    continue
                if source_id in desired_source_ids:
                    continue
                from qdrant_client import models

                self._qdrant().delete(
                    self.collection,
                    models.Filter(must=[
                        models.FieldCondition(key="knowledge_base", match=models.MatchValue(value=base)),
                        models.FieldCondition(key="source_id", match=models.MatchValue(value=source_id)),
                    ]),
                    wait=True,
                )
                catalog.execute(
                    "DELETE FROM sources WHERE knowledge_base=? AND source_id=?",
                    (base, source_id),
                )
                pruned.append(str(source_path))
            catalog.commit()
        result["manifest_documents"] = len(documents)
        result["pruned_from_manifest"] = pruned
        return result


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def get_knowledge_read_tools(config: ToolsConfig) -> list[Any]:
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")
    store = KnowledgeStore(config)

    def _search(
        query: str,
        knowledge_base: str,
        vector_limit: int,
        exact_limit: int,
        max_chars: int,
    ) -> str:
        if not knowledge_can_read(config.knowledge_mode):
            return "ERROR: knowledge search is disabled for this research request."
        try:
            return run_with_tool_logging(
                "knowledge_search",
                {
                    "query": query,
                    "knowledge_base": knowledge_base,
                    "vector_limit": vector_limit,
                    "exact_limit": exact_limit,
                },
                lambda: _json(
                    store.search(
                        query,
                        knowledge_base,
                        vector_limit=vector_limit,
                        exact_limit=exact_limit,
                        max_chars=max_chars,
                    )
                ),
            )
        except Exception as exc:
            return f"ERROR: knowledge_search failed ({type(exc).__name__}: {exc})"

    bound_base = normalize_knowledge_base(config.knowledge_base)
    if config.knowledge_bind_configured_base and bound_base:

        @function_tool(name_override="knowledge_search")
        def knowledge_search_bound(
            query: str,
            vector_limit: int = 0,
            exact_limit: int = 0,
            max_chars: int = 0,
        ) -> str:
            """Search the curated corpus selected for this research request. The corpus is policy-bound and cannot be changed by the agent.

            Args:
                query: Text or exact terms to retrieve.
                vector_limit: Dense-vector result count; zero uses, and larger values are capped by, the configured limit.
                exact_limit: Sparse lexical result count; zero uses, and larger values are capped by, the configured limit.
                max_chars: Maximum total returned passage characters; zero uses, and larger values are capped by, the configured limit.
            """
            return _search(query, bound_base, vector_limit, exact_limit, max_chars)

        @function_tool(name_override="knowledge_status")
        def knowledge_status_bound() -> str:
            """Return counts and retrieval settings for the policy-bound curated corpus."""
            try:
                return _json(store.status(bound_base))
            except Exception as exc:
                return f"ERROR: knowledge_status failed ({type(exc).__name__}: {exc})"

        return [knowledge_search_bound, knowledge_status_bound]

    @function_tool(name_override="knowledge_search")
    def knowledge_search(
        query: str,
        knowledge_base: str = "",
        vector_limit: int = 0,
        exact_limit: int = 0,
        max_chars: int = 0,
    ) -> str:
        """Search curated research using dense and lexical retrieval. Parameters: query, corpus, branch limits, and budget. Output: JSON with fused ranked chunks, provenance, rank metadata, and active limits.

        Args:
            query: Text or exact terms to retrieve.
            knowledge_base: Allowed logical corpus; empty uses the profile default.
            vector_limit: Dense-vector result count; zero uses, and larger values are capped by, the configured limit.
            exact_limit: Sparse lexical result count; zero uses, and larger values are capped by, the configured limit.
            max_chars: Maximum total returned passage characters; zero uses, and larger values are capped by, the configured limit.
        """
        return _search(query, knowledge_base, vector_limit, exact_limit, max_chars)

    @function_tool(name_override="knowledge_status")
    def knowledge_status(knowledge_base: str = "") -> str:
        """Return read-only counts and retrieval configuration. Parameters: optional allowed corpus. Output: JSON with source/chunk counts and active settings.

        Args:
            knowledge_base: Allowed logical corpus; empty uses the profile default.
        """
        try:
            return _json(store.status(knowledge_base))
        except Exception as exc:
            return f"ERROR: knowledge_status failed ({type(exc).__name__}: {exc})"

    return [knowledge_search, knowledge_status]


def get_knowledge_ingest_tools(config: ToolsConfig) -> list[Any]:
    if function_tool is None:
        raise RuntimeError("OpenAI Agents SDK is not available.")
    store = KnowledgeStore(config)

    @function_tool(name_override="knowledge_reindex")
    def knowledge_reindex(knowledge_base: str = "") -> str:
        """Incrementally embed the configured curated manifest. Parameters: optional allowed corpus. Output: JSON listing indexed, unchanged, deleted, and skipped sources.

        Args:
            knowledge_base: Allowed logical corpus; empty uses the profile default.
        """
        if not knowledge_can_write(config.knowledge_mode):
            return "ERROR: knowledge writes are disabled for this MCP profile."
        try:
            return run_with_tool_logging(
                "knowledge_reindex",
                {"knowledge_base": knowledge_base},
                lambda: _json(store.ingest_seed_manifest(knowledge_base)),
            )
        except Exception as exc:
            return f"ERROR: knowledge_reindex failed ({type(exc).__name__}: {exc})"

    return [knowledge_reindex]


def add_knowledge_read_tools(tools: list[Any], config: ToolsConfig) -> None:
    if config.knowledge_enabled and knowledge_can_read(config.knowledge_mode):
        tools.extend(get_knowledge_read_tools(config))


def add_knowledge_mcp_tools(tools: list[Any], config: ToolsConfig) -> None:
    if not config.knowledge_enabled:
        return
    tools.extend(get_knowledge_read_tools(config))
    if knowledge_can_write(config.knowledge_mode):
        tools.extend(get_knowledge_ingest_tools(config))


def documents_from_admin_decisions(
    evidence_dir: str,
    decisions: Iterable[Mapping[str, Any]],
    *,
    nominated_filenames: Iterable[str] | None = None,
    delete_verified_text_artifacts: bool = False,
    delete_verified_extracted_artifacts: bool = False,
) -> list[KnowledgeDocument]:
    root = Path(evidence_dir).expanduser().resolve()
    nominations = None if nominated_filenames is None else {
        Path(str(item)).as_posix().lstrip("./")
        for item in nominated_filenames
        if str(item).strip()
    }
    documents: list[KnowledgeDocument] = []
    for decision in decisions:
        disposition = str(decision.get("disposition") or "").strip().lower()
        if disposition != "approved":
            continue
        rel = str(decision.get("filename") or "").strip()
        if not rel:
            continue
        normalized_rel = Path(rel).as_posix().lstrip("./")
        if nominations is not None and normalized_rel not in nominations:
            continue
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        source_url = str(decision.get("source_url") or "").strip()
        stable_source_path = (
            f"source-url/{hashlib.sha256(source_url.encode('utf-8')).hexdigest()}"
            if source_url
            else f"research-artifact/{Path(rel).as_posix()}"
        )
        documents.append(KnowledgeDocument(
            path=path,
            source_path=stable_source_path,
            title=str(decision.get("title") or path.stem),
            source_url=source_url,
            evidence_type=str(decision.get("evidence_type") or "research_artifact"),
            source_group="agent-approved",
            disposition="approved",
            delete_after_ingest=(
                path.suffix.lower() in DELETABLE_TEXT_EXTENSIONS
                and (
                    bool(decision.get("delete_after_ingest"))
                    or bool(delete_verified_text_artifacts)
                )
            ),
            delete_extracted_after_ingest=bool(delete_verified_extracted_artifacts),
        ))
    return documents


def normalize_admin_knowledge_decisions(
    decisions: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Deterministically demote redundant approved representations.

    Agents still decide whether evidence is useful. Runtime owns the mechanical
    invariant that one URL maps to one retrievable source and that an extracted
    text sibling wins over equivalent markup.
    """

    rows = [dict(decision) for decision in decisions if isinstance(decision, Mapping)]

    def preference(row: Mapping[str, Any]) -> tuple[int, str]:
        filename = Path(str(row.get("filename") or ""))
        suffix_rank = {
            ".txt": 0,
            ".md": 1,
            ".markdown": 1,
            ".json": 2,
            ".jsonl": 2,
            ".csv": 3,
            ".tsv": 3,
            ".xml": 8,
            ".html": 9,
            ".htm": 9,
        }.get(filename.suffix.lower(), 5)
        return suffix_rank, filename.as_posix()

    def demote(index: int, reason: str) -> None:
        rows[index]["disposition"] = "archive_only"
        rows[index]["delete_after_ingest"] = False
        prior = str(rows[index].get("reason") or "").strip()
        rows[index]["reason"] = f"{reason} {prior}".strip()

    approved = [
        index for index, row in enumerate(rows)
        if str(row.get("disposition") or "").strip().lower() == "approved"
    ]
    by_url: dict[str, list[int]] = {}
    for index in approved:
        source_url = str(rows[index].get("source_url") or "").strip()
        if source_url:
            by_url.setdefault(source_url, []).append(index)
    for source_url, indices in by_url.items():
        if len(indices) < 2:
            continue
        winner = min(indices, key=lambda index: preference(rows[index]))
        for index in indices:
            if index != winner:
                demote(index, f"Runtime archived a redundant representation of {source_url}; the preferred representation is indexed.")

    # Some scientific tools emit same-stem XML and normalized TXT without a URL.
    # Keep XML for provenance and index the cleaner retrieval representation.
    approved = [
        index for index, row in enumerate(rows)
        if str(row.get("disposition") or "").strip().lower() == "approved"
    ]
    text_siblings = {
        (Path(str(rows[index].get("filename") or "")).parent,
         Path(str(rows[index].get("filename") or "")).stem)
        for index in approved
        if Path(str(rows[index].get("filename") or "")).suffix.lower() == ".txt"
    }
    for index in approved:
        filename = Path(str(rows[index].get("filename") or ""))
        if filename.suffix.lower() in {".xml", ".html", ".htm"} and (filename.parent, filename.stem) in text_siblings:
            demote(index, "Runtime archived redundant markup because its normalized text sibling is indexed.")
    return rows


def nominated_filenames_from_responses(responses: Iterable[Mapping[str, Any]]) -> set[str]:
    return qualified_nominated_filenames_from_responses(responses)


def qualified_nominated_filenames_from_responses(
    responses: Iterable[Mapping[str, Any]],
    *,
    evidence_dir: str = "",
) -> set[str]:
    """Return canonical candidate paths, qualified to the administrator root.

    Without ``evidence_dir`` this preserves the historical unqualified result.
    With it, each child's ``evidence_data_path`` disambiguates repeated browser
    runs that legitimately nominate the same leaf filenames.
    """

    root = Path(evidence_dir).expanduser().resolve() if str(evidence_dir).strip() else None
    filenames: set[str] = set()
    for response in responses:
        prefix = Path()
        if root is not None:
            child_root_raw = str(response.get("evidence_data_path") or "").strip()
            if not child_root_raw:
                continue
            child_root = Path(child_root_raw).expanduser().resolve()
            try:
                prefix = child_root.relative_to(root)
            except ValueError:
                continue
        candidates = response.get("knowledge_candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            raw_filename = str(candidate.get("filename") or "").strip()
            if not raw_filename:
                continue
            candidate_path = Path(raw_filename)
            if candidate_path.is_absolute() or ".." in candidate_path.parts:
                continue
            # Runtime-owned audit metadata is not research evidence. A child may
            # mention it defensively, but it must never enter the administrator's
            # exact evidence-decision set or become eligible for ingestion.
            if candidate_path.name == "_artifact_manifest.jsonl":
                continue
            filename = (prefix / candidate_path).as_posix().lstrip("./")
            if filename:
                filenames.add(filename)
    return filenames


def qualified_knowledge_candidates_from_responses(
    responses: Iterable[Mapping[str, Any]],
    *,
    evidence_dir: str = "",
) -> dict[str, dict[str, Any]]:
    """Return qualified candidate metadata keyed by administrator-relative path."""

    root = Path(evidence_dir).expanduser().resolve() if str(evidence_dir).strip() else None
    qualified: dict[str, dict[str, Any]] = {}
    for response in responses:
        prefix = Path()
        if root is not None:
            child_root_raw = str(response.get("evidence_data_path") or "").strip()
            if not child_root_raw:
                continue
            child_root = Path(child_root_raw).expanduser().resolve()
            try:
                prefix = child_root.relative_to(root)
            except ValueError:
                continue
        candidates = response.get("knowledge_candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            raw_filename = str(candidate.get("filename") or "").strip()
            if not raw_filename:
                continue
            candidate_path = Path(raw_filename)
            if (
                candidate_path.is_absolute()
                or ".." in candidate_path.parts
                or candidate_path.name == "_artifact_manifest.jsonl"
            ):
                continue
            filename = (prefix / candidate_path).as_posix().lstrip("./")
            if filename:
                row = dict(candidate)
                row["filename"] = filename
                qualified.setdefault(filename, row)
    return qualified


def validate_admin_knowledge_decisions(
    decisions: Iterable[Mapping[str, Any]],
    *,
    nominated_filenames: Iterable[str],
) -> dict[str, list[str]]:
    """Require exactly one administrator decision for every nominated path."""

    expected = {
        Path(str(item)).as_posix().lstrip("./")
        for item in nominated_filenames
        if str(item).strip()
    }
    rows = [
        Path(str(decision.get("filename") or "")).as_posix().lstrip("./")
        for decision in decisions
        if isinstance(decision, Mapping) and str(decision.get("filename") or "").strip()
    ]
    counts: dict[str, int] = {}
    for filename in rows:
        counts[filename] = counts.get(filename, 0) + 1
    actual = set(rows)
    return {
        "missing": sorted(expected - actual),
        "unexpected": sorted(actual - expected),
        "duplicates": sorted(filename for filename, count in counts.items() if count != 1),
    }


def reconcile_admin_knowledge_decisions(
    decisions: Iterable[Mapping[str, Any]],
    *,
    candidates: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Safely reconcile a model's curation rows with runtime-owned candidates.

    Research administrators sometimes shorten generated filenames or omit rows
    after a large evidence run.  A curation formatting error must not turn good
    research into a failed run, but it must also never make an invented path
    eligible for ingestion or deletion.  Runtime therefore accepts exact paths,
    canonicalizes only unambiguous provenance aliases, ignores all other
    unexpected rows, and fills every omission with ``archive_only``.
    """

    expected = {
        Path(str(filename)).as_posix().lstrip("./"): dict(candidate)
        for filename, candidate in candidates.items()
        if str(filename).strip()
    }

    def family(filename: str) -> str:
        first = Path(filename).parts[0].lower() if Path(filename).parts else ""
        for prefix in ("deepchatgpt", "prochatgpt", "websearcher", "scientific"):
            if first.startswith(prefix):
                return prefix
        return first

    def normalized_url(value: Any) -> str:
        return str(value or "").strip().rstrip("/").casefold()

    by_provenance: dict[tuple[str, str, str], list[str]] = {}
    for filename, candidate in expected.items():
        source_url = normalized_url(candidate.get("source_url"))
        if source_url:
            key = (family(filename), source_url, Path(filename).suffix.lower())
            by_provenance.setdefault(key, []).append(filename)

    canonical_rows: dict[str, list[dict[str, Any]]] = {}
    canonicalized_aliases: list[dict[str, str]] = []
    ignored_unexpected: list[str] = []
    ignored_internal: list[str] = []
    for raw_decision in decisions:
        if not isinstance(raw_decision, Mapping):
            continue
        row = dict(raw_decision)
        supplied = Path(str(row.get("filename") or "")).as_posix().lstrip("./")
        if not supplied:
            continue
        canonical = supplied if supplied in expected else ""
        if not canonical and Path(supplied).name == "_artifact_manifest.jsonl":
            ignored_internal.append(supplied)
            continue
        if not canonical:
            source_url = normalized_url(row.get("source_url"))
            if source_url:
                matches = by_provenance.get(
                    (family(supplied), source_url, Path(supplied).suffix.lower()),
                    [],
                )
                if len(matches) == 1:
                    canonical = matches[0]
        if not canonical and Path(supplied).name == "chatgpt-response.md":
            parent = Path(supplied).parent.as_posix()
            matches = [
                filename
                for filename in expected
                if Path(filename).parent.as_posix() == parent
                and re.fullmatch(r"chatgpt-(?:deep|pro)-response\.md", Path(filename).name)
            ]
            if len(matches) == 1:
                canonical = matches[0]
        if not canonical:
            ignored_unexpected.append(supplied)
            continue
        if canonical != supplied:
            canonicalized_aliases.append({"from": supplied, "to": canonical})
            row["filename"] = canonical
            candidate_url = str(expected[canonical].get("source_url") or "").strip()
            if candidate_url:
                row["source_url"] = candidate_url
        canonical_rows.setdefault(canonical, []).append(row)

    reconciled: list[dict[str, Any]] = []
    archive_only_duplicates: list[str] = []
    for filename, rows in canonical_rows.items():
        if len(rows) == 1:
            reconciled.append(rows[0])
            continue
        candidate = expected[filename]
        archive_only_duplicates.append(filename)
        reconciled.append({
            "filename": filename,
            "disposition": "archive_only",
            "reason": (
                "Runtime fail-safe archived conflicting duplicate administrator decisions; "
                "the artifact was not automatically approved or deleted."
            ),
            "title": str(candidate.get("title") or Path(filename).stem),
            "source_url": str(candidate.get("source_url") or ""),
            "evidence_type": str(candidate.get("evidence_type") or "research_artifact"),
            "delete_after_ingest": False,
        })

    archive_only_missing: list[str] = []
    decided = {str(row.get("filename") or "") for row in reconciled}
    for filename in sorted(set(expected) - decided):
        candidate = expected[filename]
        reconciled.append({
            "filename": filename,
            "disposition": "archive_only",
            "reason": (
                "Runtime fail-safe archived a researcher nomination omitted by "
                "the administrator; it was not automatically approved or deleted."
            ),
            "title": str(candidate.get("title") or Path(filename).stem),
            "source_url": str(candidate.get("source_url") or ""),
            "evidence_type": str(candidate.get("evidence_type") or "research_artifact"),
            "delete_after_ingest": False,
        })
        archive_only_missing.append(filename)

    reconciliation: dict[str, Any] = {}
    if canonicalized_aliases:
        reconciliation["canonicalized_aliases"] = canonicalized_aliases
    if ignored_unexpected:
        reconciliation["ignored_unexpected"] = sorted(set(ignored_unexpected))
    if ignored_internal:
        reconciliation["ignored_internal_metadata"] = sorted(set(ignored_internal))
    if archive_only_duplicates:
        reconciliation["archive_only_duplicates"] = sorted(archive_only_duplicates)
    if archive_only_missing:
        reconciliation["archive_only_missing"] = archive_only_missing
    return reconciled, reconciliation


def delete_discarded_artifacts(
    evidence_dir: str,
    decisions: Iterable[Mapping[str, Any]],
    *,
    nominated_filenames: Iterable[str],
) -> dict[str, list[dict[str, str]] | list[str]]:
    """Apply explicit discard decisions within one owned evidence directory."""

    root = Path(evidence_dir).expanduser().resolve()
    nominations = {Path(str(item)).as_posix().lstrip("./") for item in nominated_filenames if str(item).strip()}
    protected = {"admin_output.json", "knowledge_decisions.json", "_artifact_manifest.jsonl"}
    deleted: list[str] = []
    skipped: list[dict[str, str]] = []
    for decision in decisions:
        if str(decision.get("disposition") or "").strip().lower() != "discard":
            continue
        rel = Path(str(decision.get("filename") or "")).as_posix().lstrip("./")
        if not rel or Path(rel).name in protected:
            skipped.append({"path": rel, "reason": "protected_or_empty"})
            continue
        if not any(rel == nomination or rel.endswith(f"/{nomination}") for nomination in nominations):
            skipped.append({"path": rel, "reason": "not_nominated_by_researcher"})
            continue
        path = (root / rel).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            skipped.append({"path": rel, "reason": "outside_evidence_root"})
            continue
        if not path.is_file():
            skipped.append({"path": rel, "reason": "file_not_found"})
            continue
        path.unlink()
        _remove_artifact_manifest_entries(path, stop=root)
        deleted.append(rel)
    return {"deleted": deleted, "skipped": skipped}
