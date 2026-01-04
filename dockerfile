FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# pip 최신화
RUN pip install --upgrade pip

# 핵심 Python deps (고정 권장)
RUN pip install \
    vllm==0.3.3 \
    flash-attn==2.5.6 --no-build-isolation \
    redis

# 기본 쉘
CMD ["/bin/bash"]
