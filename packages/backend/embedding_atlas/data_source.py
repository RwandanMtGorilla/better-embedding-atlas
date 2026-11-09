# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import json
import logging
import os
import zipfile
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from weakref import finalize

import pandas as pd

from .utils import cache_path, to_parquet_bytes

logger = logging.getLogger(__name__)


def _discard_parquet(path: Path):
    try:
        path.unlink(missing_ok=True)
    except OSError as error:
        logger.warning("Could not remove temporary Parquet file %s: %s", path, error)


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
        try:
            self.cache_path = cache_path("cache", self.identifier)
        except OSError as error:
            logger.warning("Viewer cache unavailable: %s", error)
            self.cache_path = None
        self._parquet_lock = Lock()
        self._parquet_path: Path | None = None
        # ChromaDB configuration for vector search
        # Structure: {"host": str, "port": int, "collection": str, "id_to_row_index": dict[str, int]}
        self.chroma_config = chroma_config

    def cache_set(self, name: str, data):
        if self.cache_path is None:
            return
        path = self.cache_path / name
        with open(path, "w") as f:
            json.dump(data, f)

    def cache_get(self, name: str):
        if self.cache_path is None:
            return None
        path = self.cache_path / name
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        else:
            return None

    def parquet_file(self) -> Path:
        """Create one immutable Parquet file for this revision on demand."""
        with self._parquet_lock:
            if self._parquet_path is None:
                path = None
                try:
                    with NamedTemporaryFile(prefix="atlas-parquet-", suffix=".parquet", delete=False) as output:
                        path = Path(output.name)
                        self.dataset.to_parquet(output)
                except BaseException:
                    if path is not None:
                        _discard_parquet(path)
                    raise
                self._parquet_path = path
                finalize(self, _discard_parquet, path)
            return self._parquet_path

    def make_archive(self, static_path: str, *, reuse_parquet: bool = False):
        io = BytesIO()
        with zipfile.ZipFile(io, "w", zipfile.ZIP_DEFLATED) as zip:
            zip.writestr(
                "data/metadata.json",
                json.dumps(
                    self.metadata
                    | {"isStatic": True, "database": {"type": "wasm", "load": True}}
                ),
            )
            if reuse_parquet:
                zip.write(self.parquet_file(), "data/dataset.parquet")
            else:
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
            for root, _, files in (() if self.cache_path is None else os.walk(self.cache_path)):
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

            # macOS opens .command files in Terminal when they are double-clicked.
            # Finder does not guarantee that the script's directory is the current
            # working directory, so enter it explicitly before serving the files.
            start_command = """\
#!/bin/bash
cd -- "$(dirname -- "$0")" || exit 1

echo "Starting Embedding Atlas..."
echo

if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "[Error] Python not found. Please install Python first."
    read -r -p "Press Enter to close..."
    exit 1
fi

PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Server starting at: http://localhost:5051"
echo "Press Ctrl+C to stop."
echo

open http://localhost:5051
"$PYTHON_CMD" -m http.server 5051
"""
            command_info = zipfile.ZipInfo("start.command")
            command_info.create_system = 3  # Unix, so executable mode is preserved.
            command_info.external_attr = 0o100755 << 16
            zip.writestr(
                command_info,
                start_command,
                compress_type=zipfile.ZIP_DEFLATED,
            )

        return io.getvalue()
