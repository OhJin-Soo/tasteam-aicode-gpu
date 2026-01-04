# 베이스 이미지
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Flash Attention-2 설치
RUN pip install flash-attn --no-build-isolation

# vLLM 설치 (Phase 2)
RUN pip install vllm

# Redis 클라이언트 (Phase 2)
RUN pip install redis

# 애플리케이션 코드 복사
COPY . .

# 포트 노출
EXPOSE 8000

# 애플리케이션 실행
CMD ["python", "app.py"]