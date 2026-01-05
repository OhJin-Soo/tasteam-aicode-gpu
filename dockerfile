FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
# 시스템 의존성

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

# pip 최신화
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 핵심 Python deps (고정 권장)
RUN pip install \
    vllm==0.3.3 \
    flash-attn==2.5.6 --no-build-isolation \
    redis

# Set Hugging Face cache directory
ENV HF_HOME=/app/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# Pre-download model during build
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='Dilwolf/Kakao_app-kr_sentiment')"
RUN python -c "from transformers import pipeline; pipeline('sentence-similarity', model='jhgan/ko-sbert-multitask')"

COPY . /app

COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh
CMD ["/app/run.sh"]
