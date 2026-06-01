

```shell
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

# 回到 backend 运行 
cd ..\backend 
python -m embedding_atlas.cli ..\..\data\producthunt_results.csv --static ..\viewer\dist
python -m embedding_atlas.cli C:\Users\15848\Downloads\user_profiles_v2.csv --static ..\viewer\dist

uv run --active embedding-atlas  --static packages/viewer/dist --host 0.0.0.0 --port 15055

uv pip install sse-starlette
```

后端配置位于 packages/backend/.env.example
