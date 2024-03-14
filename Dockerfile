FROM python:3.11-alpine3.18

WORKDIR /app

COPY . /app

RUN apk update && \
    apk add --no-cache aws-cli ffmpeg libsm6 libxext6 unzip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]
