FROM python:3.10-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the whole repo (includes submission/ and methods/)
COPY . /app

# Optional: place prototype inside image under /app/assets/
# If you do this, copy your prototype file into assets/ before build.
# ENV PROTO_PKL=/app/assets/100k_kmean.pkl

CMD ["python3", "-m", "submission.main", "--help"]
