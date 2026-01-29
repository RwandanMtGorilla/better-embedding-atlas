# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import asyncio
import concurrent.futures
import json
import os
import re
import uuid
from functools import lru_cache
from typing import TYPE_CHECKING, Callable

import duckdb
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .data_source import DataSource
from .utils import arrow_to_bytes, to_parquet_bytes

if TYPE_CHECKING:
    from .data_source_manager import DataSourceManager, LoadingStatus


def make_server(
    data_source: DataSource,
    static_path: str,
    duckdb_uri: str | None = None,
):
    """Creates a server for hosting Embedding Atlas"""

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    mount_bytes(
        app,
        "/data/dataset.parquet",
        "application/octet-stream",
        lambda: to_parquet_bytes(data_source.dataset),
    )

    @app.get("/data/metadata.json")
    async def get_metadata():
        if duckdb_uri is None or duckdb_uri == "wasm":
            db_meta = {"database": {"type": "wasm", "load": True}}
        elif duckdb_uri == "server":
            # Point to the server itself.
            db_meta = {"database": {"type": "rest"}}
        else:
            # Point to the given uri.
            if duckdb_uri.startswith("http"):
                db_meta = {
                    "database": {"type": "rest", "uri": duckdb_uri, "load": True}
                }
            elif duckdb_uri.startswith("ws"):
                db_meta = {
                    "database": {"type": "socket", "uri": duckdb_uri, "load": True}
                }
            else:
                raise ValueError("invalid DuckDB uri")
        return data_source.metadata | db_meta

    @app.post("/data/cache/{name}")
    async def post_cache(request: Request, name: str):
        data_source.cache_set(name, await request.json())

    @app.get("/data/cache/{name}")
    async def get_cache(name: str):
        obj = data_source.cache_get(name)
        if obj is None:
            return Response(status_code=404)
        return obj

    @app.get("/data/archive.zip")
    async def make_archive():
        data = data_source.make_archive(static_path)
        return Response(content=data, media_type="application/zip")

    if duckdb_uri == "server":
        duckdb_connection = make_duckdb_connection(data_source.dataset)
    else:
        duckdb_connection = None

    def handle_query(query: dict):
        assert duckdb_connection is not None
        sql = query["sql"]
        command = query["type"]
        with duckdb_connection.cursor() as cursor:
            try:
                result = cursor.execute(sql)
                if command == "exec":
                    return JSONResponse({})
                elif command == "arrow":
                    buf = arrow_to_bytes(result.arrow())
                    return Response(
                        buf, headers={"Content-Type": "application/octet-stream"}
                    )
                elif command == "json":
                    data = result.df().to_json(orient="records")
                    return Response(data, headers={"Content-Type": "application/json"})
                else:
                    raise ValueError(f"Unknown command {command}")
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

    def handle_selection(query: dict):
        assert duckdb_connection is not None
        predicate = query.get("predicate", None)
        format = query["format"]
        formats = {
            "json": "(FORMAT JSON, ARRAY true)",
            "jsonl": "(FORMAT JSON)",
            "csv": "(FORMAT CSV)",
            "parquet": "(FORMAT parquet)",
        }
        with duckdb_connection.cursor() as cursor:
            filename = ".selection-" + str(uuid.uuid4()) + ".tmp"
            try:
                if predicate is not None:
                    cursor.execute(
                        f"COPY (SELECT * FROM dataset WHERE {predicate}) TO '{filename}' {formats[format]}"
                    )
                else:
                    cursor.execute(f"COPY dataset TO '{filename}' {formats[format]}")
                with open(filename, "rb") as f:
                    buffer = f.read()
                    return Response(
                        buffer, headers={"Content-Type": "application/octet-stream"}
                    )
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
            finally:
                try:
                    os.unlink(filename)
                except Exception:
                    pass

    executor = concurrent.futures.ThreadPoolExecutor()

    @app.get("/data/query")
    async def get_query(req: Request):
        data = json.loads(req.query_params["query"])
        return await asyncio.get_running_loop().run_in_executor(
            executor, lambda: handle_query(data)
        )

    @app.post("/data/query")
    async def post_query(req: Request):
        body = await req.body()
        data = json.loads(body)
        return await asyncio.get_running_loop().run_in_executor(
            executor, lambda: handle_query(data)
        )

    @app.post("/data/selection")
    async def post_selection(req: Request):
        body = await req.body()
        data = json.loads(body)
        return await asyncio.get_running_loop().run_in_executor(
            executor, lambda: handle_selection(data)
        )

    # Vector search endpoint (only available when ChromaDB is configured)
    if data_source.chroma_config is not None:
        from .chroma_client import ChromaDBClient, ChromaDBError
        from .embedding_service import (
            EmbeddingConfig,
            EmbeddingService,
            EmbeddingServiceError,
        )

        chroma_client = ChromaDBClient(
            host=data_source.chroma_config["host"],
            port=data_source.chroma_config["port"],
        )
        collection_name = data_source.chroma_config["collection"]
        id_to_row_index = data_source.chroma_config["id_to_row_index"]

        # Initialize embedding service if configured
        embedding_service = None
        try:
            embedding_config = EmbeddingConfig.from_env()
            embedding_service = EmbeddingService(embedding_config)
        except EmbeddingServiceError:
            pass  # Embedding service not configured, vector search will return error

        def handle_vector_search(query_data: dict):
            if embedding_service is None:
                return JSONResponse(
                    {"error": "Embedding service not configured", "code": "CONFIGURATION_ERROR"},
                    status_code=500,
                )

            query = query_data.get("query", "")
            limit = query_data.get("limit", 100)

            if not query:
                return JSONResponse(
                    {"error": "Query cannot be empty", "code": "INVALID_REQUEST"},
                    status_code=400,
                )

            try:
                # 1. Embed the query text
                query_embedding = embedding_service.embed_query(query)

                # 2. Search in ChromaDB
                search_result = chroma_client.vector_search(
                    collection_name=collection_name,
                    query_embedding=query_embedding,
                    n_results=limit,
                )

                # 3. Convert ChromaDB IDs to row indices
                results = []
                for chroma_id, distance in zip(search_result.ids, search_result.distances):
                    row_index = id_to_row_index.get(chroma_id)
                    if row_index is not None:
                        results.append({
                            "id": row_index,
                            "distance": round(distance, 5),
                        })

                return JSONResponse({"results": results})

            except EmbeddingServiceError as e:
                return JSONResponse(
                    {"error": str(e), "code": "EMBEDDING_API_ERROR"},
                    status_code=500,
                )
            except ChromaDBError as e:
                return JSONResponse(
                    {"error": str(e), "code": "CHROMADB_ERROR"},
                    status_code=500,
                )
            except Exception as e:
                return JSONResponse(
                    {"error": str(e), "code": "INTERNAL_ERROR"},
                    status_code=500,
                )

        @app.post("/data/vector-search")
        async def post_vector_search(req: Request):
            body = await req.body()
            data = json.loads(body)
            return await asyncio.get_running_loop().run_in_executor(
                executor, lambda: handle_vector_search(data)
            )

    # Static files for the frontend
    app.mount("/", StaticFiles(directory=static_path, html=True))

    return app


def make_duckdb_connection(df):
    con = duckdb.connect(":memory:")
    _ = df  # used in the query
    con.sql("CREATE TABLE dataset AS (SELECT * FROM df)")
    return con


def parse_range_header(request: Request, content_length: int):
    value = request.headers.get("Range")
    if value is not None:
        m = re.match(r"^ *bytes *= *([0-9]+) *- *([0-9]+) *$", value)
        if m is not None:
            r0 = int(m.group(1))
            r1 = int(m.group(2)) + 1
            if r0 < r1 and r0 <= content_length and r1 <= content_length:
                return (r0, r1)
    return None


def mount_bytes(
    app: FastAPI, url: str, media_type: str, make_content: Callable[[], bytes]
):
    @lru_cache(maxsize=1)
    def get_content() -> bytes:
        return make_content()

    @app.head(url)
    async def head(request: Request):
        content = get_content()
        bytes_range = parse_range_header(request, len(content))
        if bytes_range is None:
            length = len(content)
        else:
            length = bytes_range[1] - bytes_range[0]
        return Response(
            headers={
                "Content-Length": str(length),
                "Content-Type": media_type,
            }
        )

    @app.get(url)
    async def get(request: Request):
        content = get_content()
        bytes_range = parse_range_header(request, len(content))
        if bytes_range is None:
            return Response(content=content)
        else:
            r0, r1 = bytes_range
            result = content[r0:r1]
            return Response(
                content=result,
                headers={
                    "Content-Length": str(r1 - r0),
                    "Content-Range": f"bytes {r0}-{r1 - 1}/{len(content)}",
                    "Content-Type": media_type,
                },
                media_type=media_type,
                status_code=206,
            )


def make_multi_server(
    manager: "DataSourceManager",
    static_path: str,
    duckdb_uri: str | None = None,
):
    """Creates a server for hosting Embedding Atlas with multiple collections."""
    from sse_starlette.sse import EventSourceResponse

    from .data_source_manager import LoadingStatus

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # === API Endpoints ===

    @app.get("/api/collections")
    async def list_collections():
        """List all available ChromaDB collections."""
        return {"collections": manager.list_collections()}

    @app.get("/api/collection/{name}/status")
    async def get_collection_status(name: str):
        """Get the loading status of a collection."""
        managed = manager.get_managed(name)
        return {
            "name": name,
            "status": managed.loading_progress.status.value,
            "progress": managed.loading_progress.progress,
            "message": managed.loading_progress.message,
            "error": managed.loading_progress.error,
        }

    @app.get("/api/collection/{name}/progress")
    async def collection_progress_stream(name: str):
        """SSE endpoint for real-time loading progress updates."""

        async def event_generator():
            queue = manager.subscribe_progress(name)

            # Trigger loading (if not already started)
            asyncio.create_task(manager.get_or_load(name))

            try:
                while True:
                    progress = await queue.get()
                    managed = manager.get_managed(name)
                    yield {
                        "event": "progress",
                        "data": json.dumps({
                            **progress.to_dict(),
                            "logs": managed.logs,
                        }),
                    }
                    if progress.status in (LoadingStatus.READY, LoadingStatus.ERROR):
                        break
            finally:
                manager.unsubscribe_progress(name, queue)

        return EventSourceResponse(event_generator())

    # === Collection Data Endpoints ===

    # Per-collection DuckDB connections
    _collection_connections: dict[str, duckdb.DuckDBPyConnection] = {}
    _collection_executors: dict[str, concurrent.futures.ThreadPoolExecutor] = {}

    def get_duckdb_connection(name: str, data_source: DataSource):
        if name not in _collection_connections:
            con = duckdb.connect(":memory:")
            df = data_source.dataset
            con.sql("CREATE TABLE dataset AS (SELECT * FROM df)")
            _collection_connections[name] = con
        return _collection_connections[name]

    def get_executor(name: str):
        if name not in _collection_executors:
            _collection_executors[name] = concurrent.futures.ThreadPoolExecutor()
        return _collection_executors[name]

    @app.get("/collection/{name}/data/metadata.json")
    async def get_collection_metadata(name: str):
        """Get metadata for a collection."""
        managed = await manager.get_or_load(name)
        if managed.data_source is None:
            return JSONResponse(
                {"error": "Collection not loaded", "status": managed.loading_progress.status.value},
                status_code=503,
            )

        if duckdb_uri is None or duckdb_uri == "wasm":
            db_meta = {"database": {"type": "wasm", "load": True}}
        elif duckdb_uri == "server":
            db_meta = {"database": {"type": "rest"}}
        else:
            if duckdb_uri.startswith("http"):
                db_meta = {"database": {"type": "rest", "uri": duckdb_uri, "load": True}}
            elif duckdb_uri.startswith("ws"):
                db_meta = {"database": {"type": "socket", "uri": duckdb_uri, "load": True}}
            else:
                raise ValueError("invalid DuckDB uri")

        return managed.data_source.metadata | db_meta

    @app.get("/collection/{name}/data/dataset.parquet")
    async def get_collection_dataset(name: str):
        """Get the dataset as a Parquet file."""
        managed = await manager.get_or_load(name)
        if managed.data_source is None:
            return JSONResponse({"error": "Collection not loaded"}, status_code=503)

        content = to_parquet_bytes(managed.data_source.dataset)
        return Response(content=content, media_type="application/octet-stream")

    @app.post("/collection/{name}/data/cache/{cache_name}")
    async def post_collection_cache(request: Request, name: str, cache_name: str):
        """Set a cache value for a collection."""
        managed = await manager.get_or_load(name)
        if managed.data_source is None:
            return JSONResponse({"error": "Collection not loaded"}, status_code=503)
        managed.data_source.cache_set(cache_name, await request.json())

    @app.get("/collection/{name}/data/cache/{cache_name}")
    async def get_collection_cache(name: str, cache_name: str):
        """Get a cache value for a collection."""
        managed = await manager.get_or_load(name)
        if managed.data_source is None:
            return JSONResponse({"error": "Collection not loaded"}, status_code=503)
        obj = managed.data_source.cache_get(cache_name)
        if obj is None:
            return Response(status_code=404)
        return obj

    @app.get("/collection/{name}/data/archive.zip")
    async def get_collection_archive(name: str):
        """Export collection as a static archive."""
        managed = await manager.get_or_load(name)
        if managed.data_source is None:
            return JSONResponse({"error": "Collection not loaded"}, status_code=503)
        data = managed.data_source.make_archive(static_path)
        return Response(content=data, media_type="application/zip")

    @app.get("/collection/{name}/data/query")
    async def get_collection_query(name: str, req: Request):
        """Execute a DuckDB query (GET)."""
        managed = await manager.get_or_load(name)
        if managed.data_source is None:
            return JSONResponse({"error": "Collection not loaded"}, status_code=503)

        if duckdb_uri != "server":
            return JSONResponse({"error": "Server-side queries not enabled"}, status_code=400)

        data = json.loads(req.query_params["query"])
        executor = get_executor(name)
        connection = get_duckdb_connection(name, managed.data_source)

        def handle():
            return _handle_query(connection, data)

        return await asyncio.get_running_loop().run_in_executor(executor, handle)

    @app.post("/collection/{name}/data/query")
    async def post_collection_query(name: str, req: Request):
        """Execute a DuckDB query (POST)."""
        managed = await manager.get_or_load(name)
        if managed.data_source is None:
            return JSONResponse({"error": "Collection not loaded"}, status_code=503)

        if duckdb_uri != "server":
            return JSONResponse({"error": "Server-side queries not enabled"}, status_code=400)

        body = await req.body()
        data = json.loads(body)
        executor = get_executor(name)
        connection = get_duckdb_connection(name, managed.data_source)

        def handle():
            return _handle_query(connection, data)

        return await asyncio.get_running_loop().run_in_executor(executor, handle)

    @app.post("/collection/{name}/data/selection")
    async def post_collection_selection(name: str, req: Request):
        """Export selection data."""
        managed = await manager.get_or_load(name)
        if managed.data_source is None:
            return JSONResponse({"error": "Collection not loaded"}, status_code=503)

        if duckdb_uri != "server":
            return JSONResponse({"error": "Server-side queries not enabled"}, status_code=400)

        body = await req.body()
        data = json.loads(body)
        executor = get_executor(name)
        connection = get_duckdb_connection(name, managed.data_source)

        def handle():
            return _handle_selection(connection, data)

        return await asyncio.get_running_loop().run_in_executor(executor, handle)

    @app.post("/collection/{name}/data/vector-search")
    async def post_collection_vector_search(name: str, req: Request):
        """Perform vector search on a collection."""
        managed = await manager.get_or_load(name)
        if managed.data_source is None:
            return JSONResponse({"error": "Collection not loaded"}, status_code=503)

        if managed.data_source.chroma_config is None:
            return JSONResponse({"error": "Vector search not available"}, status_code=400)

        from .chroma_client import ChromaDBClient, ChromaDBError
        from .embedding_service import EmbeddingConfig, EmbeddingService, EmbeddingServiceError

        body = await req.body()
        query_data = json.loads(body)

        # Initialize services
        try:
            embedding_config = EmbeddingConfig.from_env()
            embedding_service = EmbeddingService(embedding_config)
        except EmbeddingServiceError:
            return JSONResponse(
                {"error": "Embedding service not configured", "code": "CONFIGURATION_ERROR"},
                status_code=500,
            )

        query = query_data.get("query", "")
        limit = query_data.get("limit", 100)

        if not query:
            return JSONResponse(
                {"error": "Query cannot be empty", "code": "INVALID_REQUEST"},
                status_code=400,
            )

        try:
            chroma_client = ChromaDBClient(
                host=managed.data_source.chroma_config["host"],
                port=managed.data_source.chroma_config["port"],
            )
            collection_name = managed.data_source.chroma_config["collection"]
            id_to_row_index = managed.data_source.chroma_config["id_to_row_index"]

            query_embedding = embedding_service.embed_query(query)
            search_result = chroma_client.vector_search(
                collection_name=collection_name,
                query_embedding=query_embedding,
                n_results=limit,
            )

            results = []
            for chroma_id, distance in zip(search_result.ids, search_result.distances):
                row_index = id_to_row_index.get(chroma_id)
                if row_index is not None:
                    results.append({"id": row_index, "distance": round(distance, 5)})

            return JSONResponse({"results": results})

        except EmbeddingServiceError as e:
            return JSONResponse(
                {"error": str(e), "code": "EMBEDDING_API_ERROR"},
                status_code=500,
            )
        except ChromaDBError as e:
            return JSONResponse(
                {"error": str(e), "code": "CHROMADB_ERROR"},
                status_code=500,
            )
        except Exception as e:
            return JSONResponse(
                {"error": str(e), "code": "INTERNAL_ERROR"},
                status_code=500,
            )

    # Static files for the frontend
    app.mount("/", StaticFiles(directory=static_path, html=True))

    return app


def _handle_query(connection: duckdb.DuckDBPyConnection, query: dict):
    """Handle a DuckDB query request."""
    sql = query["sql"]
    command = query["type"]
    with connection.cursor() as cursor:
        try:
            result = cursor.execute(sql)
            if command == "exec":
                return JSONResponse({})
            elif command == "arrow":
                buf = arrow_to_bytes(result.arrow())
                return Response(buf, headers={"Content-Type": "application/octet-stream"})
            elif command == "json":
                data = result.df().to_json(orient="records")
                return Response(data, headers={"Content-Type": "application/json"})
            else:
                raise ValueError(f"Unknown command {command}")
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)


def _handle_selection(connection: duckdb.DuckDBPyConnection, query: dict):
    """Handle a selection export request."""
    predicate = query.get("predicate", None)
    format = query["format"]
    formats = {
        "json": "(FORMAT JSON, ARRAY true)",
        "jsonl": "(FORMAT JSON)",
        "csv": "(FORMAT CSV)",
        "parquet": "(FORMAT parquet)",
    }
    with connection.cursor() as cursor:
        filename = ".selection-" + str(uuid.uuid4()) + ".tmp"
        try:
            if predicate is not None:
                cursor.execute(
                    f"COPY (SELECT * FROM dataset WHERE {predicate}) TO '{filename}' {formats[format]}"
                )
            else:
                cursor.execute(f"COPY dataset TO '{filename}' {formats[format]}")
            with open(filename, "rb") as f:
                buffer = f.read()
                return Response(buffer, headers={"Content-Type": "application/octet-stream"})
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)
        finally:
            try:
                os.unlink(filename)
            except Exception:
                pass
