FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir \
    grpcio \
    grpcio-tools \
    numpy

COPY . .
ENTRYPOINT ["python", "sample_test_selector.py"]