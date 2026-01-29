# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import json
import os
import zipfile
from io import BytesIO

import pandas as pd

from .utils import cache_path, to_parquet_bytes


class DataSource:
    def __init__(
        self,
        identifier: str,
        dataset: pd.DataFrame,
        metadata: dict,
        chroma_config: dict | None = None,
    ):
        self.identifier = identifier
        self.dataset = dataset
        self.metadata = metadata
        self.cache_path = cache_path("cache", self.identifier)
        # ChromaDB configuration for vector search
        # Structure: {"host": str, "port": int, "collection": str, "id_to_row_index": dict[str, int]}
        self.chroma_config = chroma_config

    def cache_set(self, name: str, data):
        path = self.cache_path / name
        with open(path, "w") as f:
            json.dump(data, f)

    def cache_get(self, name: str):
        path = self.cache_path / name
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        else:
            return None

    def make_archive(self, static_path: str):
        io = BytesIO()
        with zipfile.ZipFile(io, "w", zipfile.ZIP_DEFLATED) as zip:
            zip.writestr(
                "data/metadata.json",
                json.dumps(
                    self.metadata
                    | {"isStatic": True, "database": {"type": "wasm", "load": True}}
                ),
            )
            zip.writestr("data/dataset.parquet", to_parquet_bytes(self.dataset))
            for root, _, files in os.walk(static_path):
                for fn in files:
                    p = os.path.relpath(os.path.join(root, fn), static_path)
                    full_path = os.path.join(root, fn)
                    # 对 index.html 进行特殊处理，修改配置为单数据源模式
                    if p == "index.html":
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        # 替换配置为单数据源模式，使导出的应用直接显示UMAP图
                        content = content.replace(
                            '{ home: "multi-collection" }',
                            '{ home: "backend-viewer" }'
                        )
                        zip.writestr(p, content)
                    else:
                        zip.write(full_path, p)
            for root, _, files in os.walk(self.cache_path):
                for fn in files:
                    p = os.path.join(
                        "data/cache",
                        os.path.relpath(os.path.join(root, fn), str(self.cache_path)),
                    )
                    zip.write(os.path.join(root, fn), p)

            # 添加启动脚本
            start_bat = """\
@echo off
chcp 65001 >nul
echo Starting Embedding Atlas...
echo.

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo [Error] Python not found. Please install Python first.
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Server starting at: http://localhost:5051
echo Press Ctrl+C to stop.
echo.
start http://localhost:5051
python -m http.server 5051
"""
            zip.writestr("start.bat", start_bat)

            start_sh = """\
#!/bin/bash
echo "Starting Embedding Atlas..."
echo

if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[Error] Python not found. Please install Python first."
    exit 1
fi

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Server starting at: http://localhost:5051"
echo "Press Ctrl+C to stop."
echo

# Try to open browser
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5051 &
elif command -v open &> /dev/null; then
    open http://localhost:5051 &
fi

$PYTHON_CMD -m http.server 5051
"""
            zip.writestr("start.sh", start_sh)

        return io.getvalue()
