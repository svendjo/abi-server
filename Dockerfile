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

# Copy application files (server imports config + results_store + scorecard +
# trocr_reader + ctc_reader). digit-model*.tflite is the CNN fallback;
# cell-reader*.tflite is the CTC reader.
COPY digit-model0622.tflite cell-reader0627.tflite \
     server.py config.py results_store.py scorecard.py trocr_reader.py ctc_reader.py .
COPY config/ ./config/

# Select the production environment: config/aws-prod.yaml drives use_ctc, debug_crops,
# and the S3 (balut-results) results store. The run command stays `python server.py`.
ENV APP_ENV=aws-prod

# Run the server
CMD [ "python", "server.py" ]
