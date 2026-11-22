FROM python:3.11-slim
RUN pip install --no-cache-dir "numpy>=1.22"
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY src/ /app/src/
ENTRYPOINT ["python", "/app/src/main.py"]
