# Embedding Atlas

A Python package that provides a command line tool to visualize a dataset with embeddings. It also includes a Python Notebook (e.g., Jupyter) widget and a Streamlit widget.

- Documentation: https://apple.github.io/embedding-atlas
- GitHub: https://github.com/apple/embedding-atlas

## Installation

```bash
pip install embedding-atlas
```

and then launch the command line tool:

```bash
embedding-atlas [OPTIONS] INPUTS...
```

## Loading Data

You can load your data in two ways: locally or from Hugging Face.

### Loading Local Data

To get started with your own data, run:

```bash
embedding-atlas path_to_dataset.parquet
```

### Loading Hugging Face Data

You can instead load datasets from Hugging Face:

```bash
embedding-atlas huggingface_org/dataset_name
```

## Visualizing Embedding Projections

To visual embedding projections, pre-compute the X and Y coordinates, and specify the column names with `--x` and `--y`, such as:

```bash
embedding-atlas path_to_dataset.parquet --x projection_x --y projection_y
```

You may use the [SentenceTransformers](https://sbert.net/) package to compute high-dimensional embeddings from text data, and then use the [UMAP](https://umap-learn.readthedocs.io/en/latest/index.html) package to compute 2D projections.

### Using Pre-computed Vectors

If you already have pre-computed embedding vectors (but not the 2D projections), you can specify the column containing the vectors with `--vector`:

```bash
embedding-atlas path_to_dataset.parquet --vector embedding_vectors
```

This will apply UMAP dimensionality reduction to your pre-existing vectors without recomputing embeddings. The vectors should be stored as lists or numpy arrays in your dataset.

You may also specify a column for pre-computed nearest neighbors:

```bash
embedding-atlas path_to_dataset.parquet --x projection_x --y projection_y --neighbors neighbors
```

The `neighbors` column should have values in the following format: `{"ids": [id1, id2, ...], "distances": [d1, d2, ...]}`.
If this column is specified, you'll be able to see nearest neighbors for a selected point in the tool.

## Local Development

Launch Embedding Altas with a wine reviews dataset with `./start.sh` and the MNIST dataset with `./start_image.sh`.

```sh
cd D:\projects\better-embedding-atlas

uv venv --python 3.12
.venv\Scripts\activate

cd packages/backend
# 可编辑安装
uv pip install -e .

# 构建前端
npm install

cd packages\viewer

# 此项依赖其他包先编译 首次运行需要 在根目录build (或在windows: .\scripts\build.bat) 
npm run build 

uv run --active embedding-atlas  --static ../viewer/dist --host 0.0.0.0

uv run --active embedding-atlas  --static packages/viewer/dist --host 0.0.0.0 --port 15055

uv pip install sse-starlette
```
