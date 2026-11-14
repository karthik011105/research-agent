# 1) the web app (frontend/): React + Vite, built to static files. The Results page bundles bench/results/.
FROM node:24-slim AS web
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
COPY bench/results/ /src/bench/results/
RUN npm run build

# 2) the API, which also serves the built web app from /app/web
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only PyTorch (for sentence-transformers embeddings). The default PyPI build bundles ~3 GB of NVIDIA CUDA
# libraries that Fargate and laptops never use: slower builds and deploys, more ECR storage.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

ENV HF_HUB_OFFLINE=1

COPY app/ ./app/
COPY index.html ./index.html
COPY --from=web /src/frontend/dist ./web

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
