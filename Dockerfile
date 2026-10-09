FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY day15.py .

CMD ["uvicorn", "day15:app", "--host", "0.0.0.0", "--port", "8000"]