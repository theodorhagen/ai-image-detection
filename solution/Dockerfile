FROM python:3.11-slim

WORKDIR /workspace/solution

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt /workspace/solution/requirements.txt

RUN pip install --no-cache-dir \
    -r /workspace/solution/requirements.txt

RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 \
    torchvision==0.20.1

COPY . /workspace/solution
