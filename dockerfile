FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 시스템 의존성
RUN apt-get update --yes && \
    DEBIAN_FRONTEND=noninteractive apt-get install --yes --no-install-recommends \
        wget \
        curl \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

# pip 최신화
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 핵심 Python deps (고정 권장)
# Flash Attention-2는 선택적 (실패해도 계속 진행)
RUN pip install redis && \
    (pip install flash-attn==2.5.6 --no-build-isolation || echo "Flash Attention-2 설치 실패") && \
    pip install vllm==0.3.3

# Set Hugging Face cache directory (런타임에 덮어쓸 수 있음)
ENV HF_HOME=/workspace/models
ENV HF_HUB_ENABLE_HF_TRANSFER=0

# 모델 사전 다운로드 (선택적, 실패해도 계속 진행)
# 빌드 시간을 줄이려면 주석 처리하고 런타임에 다운로드
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='Dilwolf/Kakao_app-kr_sentiment')" || true
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('jhgan/ko-sbert-multitask')" || true
# Qwen2.5-7B-Instruct 모델 사전 다운로드 (선택적, 실패해도 계속 진행)
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    print('Qwen 모델 다운로드 시작...'); \
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True); \
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True); \
    print('Qwen 모델 다운로드 완료')" || echo "Qwen 모델 사전 다운로드 실패, 런타임에 다운로드됩니다"

COPY . /app

COPY run.sh /app/run.sh
RUN chmod +x /app/run.sh

CMD ["/app/run.sh"]