FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEMO_RUNTIME=remote
ENV HOST=0.0.0.0
ENV PORT=8080

WORKDIR /app

COPY requirements.remote.txt .
RUN pip install --no-cache-dir -r requirements.remote.txt

COPY demo.py index.html index.js ./
COPY providers ./providers

EXPOSE 8080

CMD ["python", "demo.py", "--runtime", "remote"]
