FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /app

COPY pyproject.toml uv.lock /app/

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.2 /uv /uvx /bin/

RUN uv sync --frozen