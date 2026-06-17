# The tflite-runtime 2.14 library is only available in Python 3.10 and 3.11
FROM python:3.11-slim

WORKDIR /app

# OpenCV (headless) still needs a couple of shared libs at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY digit-model0608.tflite server.py .

# Run the server
CMD [ "python", "server.py" ]
