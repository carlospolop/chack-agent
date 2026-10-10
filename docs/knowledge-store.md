# Curated research knowledge

Chack uses one local Qdrant collection with `knowledge_base` payload isolation. Qdrant is a retrieval index; a small SQLite catalog records source hashes and indexing parameters. Agents can search but cannot mutate the index. Only the MCP/runtime approval path can ingest documents.

## Request policy

Every queue or administrator request accepts:

- `knowledge_mode`: `off`, `read`, `write`, or `read_write`
- `knowledge_base`: an allow-listed logical corpus such as `lipedema`

Use `off` for disposable research, `read` when prior knowledge may be consulted but the result must not persist, and `read_write` for a continuing research program. Requests with different policies are never merged into one queue job.

## Retrieval and curation

`knowledge_search` runs dense multilingual retrieval and sparse BM25 lexical retrieval separately, then reciprocal-rank fuses them. Profile settings control chunk size/overlap, result counts for each branch, and the maximum returned text.

Ordinary MCP-capable researchers receive a request-bound, read-only `knowledge_search` tool and cannot select another corpus. Browser-backed ChatGPT Deep/Pro researchers cannot call local MCP tools from the webpage, so their wrapper performs the same policy-bound hybrid search before browser submission and injects the provenance-tagged passages into the exact request as untrusted leads. It writes `knowledge-retrieval.json` as an `archive_only` audit receipt and fails closed instead of launching the browser if the required Qdrant read fails or returns no passages. The browser preface has independent operator caps (`knowledge_browser_vector_results`, `knowledge_browser_exact_results`, and `knowledge_browser_max_return_chars`) which are also bounded by the collection-wide limits; keeping this preface smaller than an administrator's interactive search prevents catalogue metadata from crowding out the actual Deep/Pro request.

Researchers classify every retained artifact exactly once as `ingest_candidate`, `archive_only`, or `discard`; an omitted classification is deterministically downgraded to `archive_only`. The administrator reviews every nomination and returns final decisions. Runtime code validates paths and embeds only approved files. With `knowledge_delete_verified_text_artifacts=true`, approved disposable text in the run workspace is removed only after Qdrant contains a hash-verified exact extracted-source copy. The stronger per-profile `knowledge_delete_verified_extracted_artifacts=true` also removes disposable HTML, XML, and PDF downloads after the extracted text is hash-verified in Qdrant; it deliberately does not preserve their original bytes, markup, or layout. Canonical manifest/repository sources, manifests, audit receipts, and unsupported binaries are never automatically deleted.

An administrator result is fail-closed if any launched researcher remains non-terminal in its durable ledger. Failed/incomplete administrator runs never ingest their proposed decisions. Continuing programs should also configure `researcher_queue_required_researchers` for specialist families that every run must include.

For long browser research, size the queue as one explicit wall-clock budget: `researcher_queue_max_runtime_minutes` includes both child work and final curation, while `researcher_administrator_synthesis_reserve_minutes` is kept exclusively for synthesis and knowledge decisions. The queue's private administrator caps browser polling and all child supervision at the remaining researcher window, even across an MCP process boundary. Keep `researcher_queue_max_wait_seconds` and the outer MCP timeout longer than the administrator runtime.

## Operations

Start the local service:

```bash
docker compose -f docker-compose.qdrant.yml up -d
```

The configured `knowledge_reindex` MCP tool incrementally ingests a curated manifest and skips only sources whose content hash, model, FastEmbed runtime, chunk size, and overlap all match. `knowledge_status` reports compatible and incompatible source counts so a runtime upgrade cannot silently mix vector semantics. Back up both the Qdrant data directory and SQLite catalog; source repositories remain the rebuild authority for canonical documents.
