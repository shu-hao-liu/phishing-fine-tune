FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/phishing-detect
COPY requirements.txt ./
RUN python3 -m pip install --no-cache-dir -r requirements.txt
COPY finetune.py test.py analyze_results.py config.yaml ./
# Keep the Pod alive for interactive commands; no network services are exposed.
CMD ["sleep", "infinity"]
