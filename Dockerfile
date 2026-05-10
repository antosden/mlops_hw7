FROM python:3.11-slim

WORKDIR /app

RUN pip install numpy pandas scikit-learn

COPY ml_pipeline.py .

CMD ["python", "ml_pipeline.py"]