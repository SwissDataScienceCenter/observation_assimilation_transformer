FROM python:3.12-slim-bookworm

RUN apt-get update && \
    apt-get install -y git && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
# Install Poetry
RUN pip install poetry
RUN poetry config virtualenvs.in-project true
RUN mypy --install-types --non-interactive

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

    
RUN mypy --install-types --non-interactive
