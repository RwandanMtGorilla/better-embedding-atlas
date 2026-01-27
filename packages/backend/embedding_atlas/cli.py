# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Command line interface."""

import importlib
import logging
import os
import pathlib
import socket
from pathlib import Path

import click
import inquirer
import numpy as np
import pandas as pd
import uvicorn

from .data_source import DataSource
from .options import make_embedding_atlas_props
from .server import make_server
from .utils import (
    Hasher,
    cache_path,
    load_dotenv_config,
    load_huggingface_data,
    load_pandas_data,
)
from .version import __version__


def find_column_name(existing_names, candidate):
    if candidate not in existing_names:
        return candidate
    else:
        index = 1
        while True:
            s = f"{candidate}_{index}"
            if s not in existing_names:
                return s
            index += 1


def determine_and_load_data(filename: str, splits: list[str] | None = None):
    suffix = Path(filename).suffix.lower()
    hf_prefix = "hf://datasets/"

    # Override Hugging Face data if given full url
    if filename.startswith(hf_prefix):
        filename = filename.split(hf_prefix)[-1]

    # Hugging Face data
    if (len(filename.split("/")) <= 2) and (suffix == ""):
        df = load_huggingface_data(filename, splits)
    else:
        df = load_pandas_data(filename)

    return df


def query_dataframe(query: str, data: pd.DataFrame) -> pd.DataFrame:
    import duckdb

    _ = data  # used in query
    return duckdb.sql(query).df()


def load_datasets(
    inputs: list[str],
    splits: list[str] | None = None,
    query: str | None = None,
    sample: int | None = None,
) -> pd.DataFrame:
    existing_column_names = set()
    dataframes = []
    for fn in inputs:
        print("Loading data from " + fn)
        df = determine_and_load_data(fn, splits=splits)
        dataframes.append(df)
        for c in df.columns:
            existing_column_names.add(c)

    file_name_column = find_column_name(existing_column_names, "FILE_NAME")
    for df, fn in zip(dataframes, inputs):
        df[file_name_column] = fn

    df = pd.concat(dataframes)

    if query is not None:
        df = query_dataframe(query, df)

    if sample:
        df = df.sample(n=sample, axis=0, random_state=np.random.RandomState(42))

    return df


def prompt_for_column(df: pd.DataFrame, message: str) -> str | None:
    question = [
        inquirer.List(
            "arg",
            message=message,
            choices=sorted(["(none)"] + [str(c) for c in df.columns]),
        ),
    ]
    r = inquirer.prompt(question)
    if r is None:
        return None
    text = r["arg"]  # type: ignore
    if text == "(none)":
        text = None
    return text


def find_available_port(start_port: int, max_attempts: int = 10, host="localhost"):
    """Find the next available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError("No available ports found in the given range")


def import_modules(names: list[str]):
    """Import the given list of modules."""
    for name in names:
        importlib.import_module(name)


def load_from_chromadb(
    collection_name: str,
    chroma_host: str,
    chroma_port: int,
    umap_args: dict,
    query: str | None = None,
    sample: int | None = None,
) -> tuple[pd.DataFrame, str, str, str, dict[str, int]]:
    """
    Load data from ChromaDB collection with caching support.

    Args:
        collection_name: ChromaDB collection name
        chroma_host: ChromaDB server host
        chroma_port: ChromaDB server port
        umap_args: UMAP parameters
        query: Optional SQL query to filter data
        sample: Optional number of samples to draw

    Returns:
        Tuple of (dataframe, x_column, y_column, neighbors_column, chroma_id_to_row_index)
    """
    from .chroma_client import (
        ChromaDBClient,
        chroma_data_to_dataframe,
    )
    from .projection import Projection, _run_umap

    logger = logging.getLogger(__name__)

    # Initialize client
    client = ChromaDBClient(host=chroma_host, port=chroma_port)

    # Step 1: Get documents and metadata (for cache check)
    logger.info("Fetching documents and metadata from ChromaDB...")
    data = client.get_documents_and_metadata(collection_name)
    row_count = len(data.ids)

    # Step 2: Compute cache key
    hasher = Hasher()
    hasher.update(
        {
            "version": 1,
            "source": "chromadb",
            "collection": collection_name,
            "documents": data.documents,
            "row_count": row_count,
            "umap_args": umap_args,
        }
    )
    cache_key = hasher.hexdigest()
    cache_file = cache_path("projections") / cache_key

    # Step 3: Check cache
    proj = None
    if Projection.exists(cache_file):
        logger.info("Found cached projection, loading from %s", str(cache_file))
        proj = Projection.load(cache_file)

        # Validate row count
        if proj.projection.shape[0] == row_count:
            logger.info("Cache is valid, using cached projection")
        else:
            logger.warning(
                "Cache row count mismatch (%d vs %d), recomputing...",
                proj.projection.shape[0],
                row_count,
            )
            proj = None
    else:
        logger.info("No cached projection found")

    # Step 4: Compute projection if needed
    if proj is None:
        logger.info("Fetching embeddings from ChromaDB...")
        embeddings = client.get_embeddings(collection_name)

        logger.info("Running UMAP on %d embeddings...", len(embeddings))
        proj = _run_umap(embeddings, umap_args)

        # Save to cache
        Projection.save(cache_file, proj)
        logger.info("Saved projection to cache: %s", str(cache_file))

    # Step 5: Convert to DataFrame
    df = chroma_data_to_dataframe(data)

    # Step 6: Apply query and sample if specified
    if query is not None:
        df = query_dataframe(query, df)
        # Note: query may change row count, but we still use the full projection
        # This is a limitation - query should be applied before projection
        if len(df) != row_count:
            logger.warning(
                "Query changed row count from %d to %d. "
                "Projection coordinates may not match correctly. "
                "Consider using --query before loading from ChromaDB.",
                row_count,
                len(df),
            )

    if sample is not None and sample < len(df):
        # Sample rows and corresponding projection points
        indices = df.sample(
            n=sample, axis=0, random_state=np.random.RandomState(42)
        ).index.tolist()
        df = df.loc[indices].reset_index(drop=True)
        proj = Projection(
            projection=proj.projection[indices],
            knn_indices=proj.knn_indices[indices],
            knn_distances=proj.knn_distances[indices],
        )
        logger.info("Sampled %d rows from dataset", sample)

    # Step 7: Add projection columns
    x_column = find_column_name(df.columns, "projection_x")
    y_column = find_column_name(df.columns, "projection_y")
    neighbors_column = find_column_name(df.columns, "__neighbors")

    df[x_column] = proj.projection[:, 0]
    df[y_column] = proj.projection[:, 1]
    df[neighbors_column] = [
        {"distances": d.tolist(), "ids": i.tolist()}
        for i, d in zip(proj.knn_indices, proj.knn_distances)
    ]

    # Build ChromaDB ID to row index mapping
    chroma_id_to_row_index = {chroma_id: idx for idx, chroma_id in enumerate(data.ids)}

    return df, x_column, y_column, neighbors_column, chroma_id_to_row_index


@click.command()
@click.argument("inputs", nargs=-1, required=False)
@click.option("--text", default=None, help="Column containing text data.")
@click.option("--image", default=None, help="Column containing image data.")
@click.option(
    "--vector", default=None, help="Column containing pre-computed vector embeddings."
)
@click.option(
    "--split",
    default=[],
    multiple=True,
    help="Dataset split name(s) to load from Hugging Face datasets. Can be specified multiple times for multiple splits.",
)
@click.option(
    "--enable-projection/--disable-projection",
    "enable_projection",
    default=True,
    help="Compute embedding projections from text/image/vector data. If disabled without pre-computed projections, the embedding view will be unavailable.",
)
@click.option(
    "--model",
    default=None,
    help="Model name for generating embeddings (e.g., 'all-MiniLM-L6-v2').",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    default=False,
    help="Allow execution of remote code when loading models from Hugging Face Hub.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for processing embeddings (default: 32 for text, 16 for images). Larger values use more memory but may be faster.",
)
@click.option(
    "--text-projector",
    type=click.Choice(["sentence_transformers", "litellm"]),
    default="sentence_transformers",
    help="Embedding provider: 'sentence_transformers' (local) or 'litellm' (API-based).",
)
@click.option(
    "--api-key",
    type=str,
    default=None,
    help="API key for litellm embedding provider.",
)
@click.option(
    "--api-base",
    type=str,
    default=None,
    help="API endpoint for litellm embedding provider.",
)
@click.option(
    "--dimensions",
    type=int,
    default=None,
    help="Number of dimensions for output embeddings (litellm only, supported by OpenAI text-embedding-3+).",
)
@click.option(
    "--sync",
    is_flag=True,
    default=False,
    help="Process embeddings synchronously (litellm only). Use for local servers like Ollama to avoid memory issues.",
)
@click.option(
    "--x",
    "x_column",
    help="Column containing pre-computed X coordinates for the embedding view.",
)
@click.option(
    "--y",
    "y_column",
    help="Column containing pre-computed Y coordinates for the embedding view.",
)
@click.option(
    "--neighbors",
    "neighbors_column",
    help='Column containing pre-computed nearest neighbors in format: {"ids": [n1, n2, ...], "distances": [d1, d2, ...]}. IDs should be zero-based row indices.',
)
@click.option(
    "--projection-cache",
    "projection_cache",
    type=str,
    default=None,
    help="Identifier for loading pre-computed projection from cache. "
         "Loads {identifier}.projection.npy, {identifier}.knn_indices.npy, "
         "and {identifier}.knn_distances.npy from the projections cache directory.",
)
@click.option(
    "--chroma-collection",
    "chroma_collection",
    type=str,
    default=None,
    help="ChromaDB collection name to load data from. Mutually exclusive with INPUTS argument.",
)
@click.option(
    "--chroma-host",
    "chroma_host",
    type=str,
    default=None,
    help="ChromaDB server host. Overrides CHROMA_HOST environment variable. Default: localhost.",
)
@click.option(
    "--chroma-port",
    "chroma_port",
    type=int,
    default=None,
    help="ChromaDB server port. Overrides CHROMA_PORT environment variable. Default: 8000.",
)
@click.option(
    "--query",
    default=None,
    type=str,
    help="Use the result of the given SQL query as input data. In the query, you may refer to the original data as 'data'.",
)
@click.option(
    "--sample",
    default=None,
    type=int,
    help="Number of random samples to draw from the dataset. Useful for large datasets. If query is specified, sampling applies after the query.",
)
@click.option(
    "--umap-n-neighbors",
    type=int,
    help="Number of neighbors to consider for UMAP dimensionality reduction (default: 15).",
)
@click.option(
    "--umap-min-dist",
    type=float,
    help="The min_dist parameter for UMAP.",
)
@click.option(
    "--umap-metric",
    default="cosine",
    help="Distance metric for UMAP computation (default: 'cosine').",
)
@click.option(
    "--umap-random-state", type=int, help="Random seed for reproducible UMAP results."
)
@click.option(
    "--duckdb",
    type=str,
    default="server",
    help="DuckDB connection mode: 'wasm' (run in browser), 'server' (run on this server), or URI (e.g., 'ws://localhost:3000').",
)
@click.option(
    "--host",
    default="localhost",
    help="Host address for the web server (default: localhost).",
)
@click.option(
    "--port", default=5055, help="Port number for the web server (default: 5055)."
)
@click.option(
    "--auto-port/--no-auto-port",
    "enable_auto_port",
    default=True,
    help="Automatically find an available port if the specified port is in use.",
)
@click.option(
    "--static", type=str, help="Custom path to frontend static files directory."
)
@click.option(
    "--export-application",
    type=str,
    help="Export the visualization as a standalone web application to the specified ZIP file and exit.",
)
@click.option(
    "--with",
    "with_modules",
    default=[],
    multiple=True,
    help="Import the given Python module before loading data. For example, you can use this to import fsspec filesystems. Can be specified multiple times to import multiple modules.",
)
@click.option(
    "--point-size",
    type=float,
    default=None,
    help="Size of points in the embedding view (default: automatically calculated based on density).",
)
@click.option(
    "--stop-words",
    type=str,
    default=None,
    help="Path to a file containing stop words to exclude from the text embedding. The file should be a table with column 'word'",
)
@click.option(
    "--labels",
    type=str,
    default=None,
    help="Path to a file containing labels for the embedding view. The file should be a table with columns 'x', 'y', 'text', and optionally 'level' and 'priority'",
)
@click.option(
    "--multi-collection",
    "multi_collection_flag",
    is_flag=True,
    default=None,
    help="Enable multi-collection mode. When enabled, all ChromaDB collections are accessible via API. Overrides MULTI_COLLECTION_MODE env var.",
)
@click.option(
    "--max-cached",
    "max_cached_arg",
    type=int,
    default=None,
    help="Maximum number of collections to keep loaded in memory (LRU cache). Overrides MAX_CACHED_COLLECTIONS env var. Default: 5.",
)
@click.version_option(version=__version__, package_name="embedding_atlas")
def main(
    inputs,
    text: str | None,
    image: str | None,
    vector: str | None,
    split: list[str] | None,
    enable_projection: bool,
    model: str | None,
    trust_remote_code: bool,
    batch_size: int | None,
    text_projector: str,
    api_key: str | None,
    api_base: str | None,
    dimensions: int | None,
    sync: bool,
    x_column: str | None,
    y_column: str | None,
    neighbors_column: str | None,
    projection_cache: str | None,
    chroma_collection: str | None,
    chroma_host: str | None,
    chroma_port: int | None,
    query: str | None,
    sample: int | None,
    umap_n_neighbors: int | None,
    umap_min_dist: int | None,
    umap_metric: str | None,
    umap_random_state: int | None,
    static: str | None,
    duckdb: str,
    host: str,
    port: int,
    enable_auto_port: bool,
    export_application: str | None,
    with_modules: list[str] | None,
    point_size: float | None,
    stop_words: str | None,
    labels: str | None,
    multi_collection_flag: bool | None,
    max_cached_arg: int | None,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: (%(name)s) %(message)s",
    )

    # Load environment variables from .env file
    load_dotenv_config()

    if with_modules is not None:
        import_modules(with_modules)

    # Read multi-collection mode settings from env or CLI args
    multi_collection = os.environ.get("MULTI_COLLECTION_MODE", "false").lower() == "true"
    max_cached = int(os.environ.get("MAX_CACHED_COLLECTIONS", "5"))

    # CLI arguments override environment variables
    if multi_collection_flag is not None:
        multi_collection = multi_collection_flag
    if max_cached_arg is not None:
        max_cached = max_cached_arg

    # Multi-collection mode
    if multi_collection:
        # Resolve ChromaDB connection parameters from env or defaults
        resolved_chroma_host = chroma_host or os.environ.get("CHROMA_HOST", "localhost")
        resolved_chroma_port = chroma_port or int(os.environ.get("CHROMA_PORT", "8000"))

        # Build UMAP args
        umap_args = {}
        if umap_min_dist is not None:
            umap_args["min_dist"] = umap_min_dist
        if umap_n_neighbors is not None:
            umap_args["n_neighbors"] = umap_n_neighbors
        if umap_random_state is not None:
            umap_args["random_state"] = umap_random_state
        if umap_metric is not None:
            umap_args["metric"] = umap_metric

        from .data_source_manager import DataSourceManager
        from .server import make_multi_server

        logging.info("Starting in multi-collection mode")
        logging.info(f"ChromaDB: {resolved_chroma_host}:{resolved_chroma_port}")
        logging.info(f"Max cached collections: {max_cached}")

        manager = DataSourceManager(
            chroma_host=resolved_chroma_host,
            chroma_port=resolved_chroma_port,
            max_cached=max_cached,
            umap_args=umap_args,
            duckdb_uri=duckdb,
        )

        if static is None:
            static = str((pathlib.Path(__file__).parent / "static").resolve())

        app = make_multi_server(manager, static_path=static, duckdb_uri=duckdb)

        if enable_auto_port:
            new_port = find_available_port(port, max_attempts=10, host=host)
            if new_port != port:
                logging.info(f"Port {port} is not available, using {new_port}")
        else:
            new_port = port

        uvicorn.run(app, port=new_port, host=host, access_log=False)
        return

    # Parameter validation: ChromaDB vs inputs are mutually exclusive
    if chroma_collection is not None and len(inputs) > 0:
        raise click.UsageError(
            "--chroma-collection cannot be used together with INPUTS argument. "
            "Please use either --chroma-collection or provide input files, not both."
        )

    if chroma_collection is None and len(inputs) == 0:
        raise click.UsageError(
            "Either INPUTS argument or --chroma-collection must be provided."
        )

    # Build UMAP args
    umap_args = {}
    if umap_min_dist is not None:
        umap_args["min_dist"] = umap_min_dist
    if umap_n_neighbors is not None:
        umap_args["n_neighbors"] = umap_n_neighbors
    if umap_random_state is not None:
        umap_args["random_state"] = umap_random_state
    if umap_metric is not None:
        umap_args["metric"] = umap_metric

    # Load data from appropriate source
    chroma_config = None
    if chroma_collection is not None:
        # Resolve ChromaDB connection parameters from env or defaults
        resolved_chroma_host = chroma_host or os.environ.get("CHROMA_HOST", "localhost")
        resolved_chroma_port = chroma_port or int(os.environ.get("CHROMA_PORT", "8000"))

        # Load from ChromaDB with error handling
        try:
            df, x_column, y_column, neighbors_column, chroma_id_to_row_index = load_from_chromadb(
                collection_name=chroma_collection,
                chroma_host=resolved_chroma_host,
                chroma_port=resolved_chroma_port,
                umap_args=umap_args,
                query=query,
                sample=sample,
            )
        except ImportError as e:
            raise click.UsageError(str(e))
        except Exception as e:
            # Handle ChromaDB errors (ChromaDBError and subclasses)
            error_name = type(e).__name__
            if "ChromaDB" in error_name:
                raise click.UsageError(str(e))
            raise

        # Build ChromaDB config for vector search
        chroma_config = {
            "host": resolved_chroma_host,
            "port": resolved_chroma_port,
            "collection": chroma_collection,
            "id_to_row_index": chroma_id_to_row_index,
        }

        # Set text column to document if not specified
        if text is None:
            text = "document"

        print(df)
    else:
        # Existing logic: load from files
        df = load_datasets(inputs, splits=split, query=query, sample=sample)

        print(df)

    # 处理 --projection-cache 参数 (only when not using ChromaDB)
    if chroma_collection is None and projection_cache is not None:
        # 参数冲突检查
        if x_column is not None or y_column is not None:
            raise click.UsageError(
                "--projection-cache cannot be used together with --x or --y."
            )

        if image is not None or vector is not None:
            raise click.UsageError(
                "--projection-cache cannot be used together with --image or --vector."
            )

        from .projection import Projection

        cache_dir = cache_path("projections")
        cache_file = cache_dir / projection_cache

        # 检查文件是否存在
        if not Projection.exists(cache_file):
            missing = []
            for suffix in [".projection.npy", ".knn_indices.npy", ".knn_distances.npy"]:
                if not cache_file.with_suffix(suffix).exists():
                    missing.append(f"{projection_cache}{suffix}")
            raise click.UsageError(
                f"Projection cache not found in {cache_dir}. Missing: {', '.join(missing)}"
            )

        logging.info("Loading projection from cache: %s", str(cache_file))
        proj = Projection.load(cache_file)

        # 验证行数
        if proj.projection.shape[0] != len(df):
            raise click.UsageError(
                f"Cache has {proj.projection.shape[0]} rows, dataset has {len(df)} rows."
            )

        # 创建列名
        x_column = find_column_name(df.columns, "projection_x")
        y_column = find_column_name(df.columns, "projection_y")
        if neighbors_column is None:
            neighbors_column = find_column_name(df.columns, "__neighbors")

        # 注入数据
        df[x_column] = proj.projection[:, 0]
        df[y_column] = proj.projection[:, 1]
        df[neighbors_column] = [
            {"distances": d.tolist(), "ids": i.tolist()}
            for i, d in zip(proj.knn_indices, proj.knn_distances)
        ]

        logging.info("Loaded projection with %d points from cache", len(df))

    elif chroma_collection is None and enable_projection and (x_column is None or y_column is None):
        # No x, y column selected, first see if text/image/vectors column is specified, if not, ask for it
        if text is None and image is None and vector is None:
            text = prompt_for_column(
                df, "Select a column you want to run the embedding on"
            )
        umap_args = {}
        if umap_min_dist is not None:
            umap_args["min_dist"] = umap_min_dist
        if umap_n_neighbors is not None:
            umap_args["n_neighbors"] = umap_n_neighbors
        if umap_random_state is not None:
            umap_args["random_state"] = umap_random_state
        if umap_metric is not None:
            umap_args["metric"] = umap_metric
        # Run embedding and projection
        if text is not None or image is not None or vector is not None:
            from .projection import (
                compute_image_projection,
                compute_text_projection,
                compute_vector_projection,
            )

            x_column = find_column_name(df.columns, "projection_x")
            y_column = find_column_name(df.columns, "projection_y")
            if neighbors_column is None:
                neighbors_column = find_column_name(df.columns, "__neighbors")
                new_neighbors_column = neighbors_column
            else:
                # If neighbors_column is already specified, don't overwrite it.
                new_neighbors_column = None
            if vector is not None:
                compute_vector_projection(
                    df,
                    vector,
                    x=x_column,
                    y=y_column,
                    neighbors=new_neighbors_column,
                    umap_args=umap_args,
                )
            elif text is not None:
                # Build kwargs for litellm projector
                litellm_kwargs = {}
                if api_key is not None:
                    litellm_kwargs["api_key"] = api_key
                if api_base is not None:
                    litellm_kwargs["api_base"] = api_base
                if dimensions is not None:
                    litellm_kwargs["dimensions"] = dimensions
                if sync:
                    litellm_kwargs["sync"] = sync

                compute_text_projection(
                    df,
                    text,
                    x=x_column,
                    y=y_column,
                    neighbors=new_neighbors_column,
                    model=model,
                    text_projector=text_projector,  # type: ignore
                    trust_remote_code=trust_remote_code,
                    batch_size=batch_size,
                    umap_args=umap_args,
                    **litellm_kwargs,
                )
            elif image is not None:
                compute_image_projection(
                    df,
                    image,
                    x=x_column,
                    y=y_column,
                    neighbors=new_neighbors_column,
                    model=model,
                    trust_remote_code=trust_remote_code,
                    batch_size=batch_size,
                    umap_args=umap_args,
                )
            else:
                raise RuntimeError("unreachable")

    id_column = find_column_name(df.columns, "__row_index__")
    df[id_column] = range(df.shape[0])

    stop_words_resolved = None
    if stop_words is not None:
        stop_words_df = load_pandas_data(stop_words)
        stop_words_resolved = stop_words_df["word"].to_list()

    labels_resolved = None
    if labels is not None:
        labels_df = load_pandas_data(labels)
        labels_resolved = labels_df.to_dict("records")

    props = make_embedding_atlas_props(
        row_id=id_column,
        x=x_column,
        y=y_column,
        neighbors=neighbors_column,
        text=text,
        point_size=point_size,
        stop_words=stop_words_resolved,
        labels=labels_resolved,
    )

    metadata = {
        "props": props,
    }

    hasher = Hasher()
    hasher.update(__version__)
    hasher.update(inputs)
    hasher.update(metadata)
    identifier = hasher.hexdigest()

    dataset = DataSource(identifier, df, metadata, chroma_config=chroma_config)

    if static is None:
        static = str((pathlib.Path(__file__).parent / "static").resolve())

    if export_application is not None:
        with open(export_application, "wb") as f:
            f.write(dataset.make_archive(static))
        exit(0)

    app = make_server(dataset, static_path=static, duckdb_uri=duckdb)

    if enable_auto_port:
        new_port = find_available_port(port, max_attempts=10, host=host)
        if new_port != port:
            logging.info(f"Port {port} is not available, using {new_port}")
    else:
        new_port = port
    uvicorn.run(app, port=new_port, host=host, access_log=False)


if __name__ == "__main__":
    main()
