# Use the official Python 3.11 image based on Alpine 3.18
FROM python:3.11-alpine3.18

# Set the working directory inside the container
WORKDIR /app

# Copy your application code into the container
COPY . /app

# Install system dependencies and Python packages
RUN apk update && \
    apk add --no-cache aws-cli ffmpeg libsm6 libxext6 unzip && \
    pip install --no-cache-dir -r requirements.txt

# Specify the default command to run your application
CMD ["python3", "app.py"]