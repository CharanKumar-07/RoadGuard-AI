FROM python:3.10-slim

WORKDIR /app

# System deps for psycopg2 (optional) and other native libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY evidence/ ./evidence/         

# Create evidence dir inside the container (bind-mounted at runtime)
RUN mkdir -p evidence

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
