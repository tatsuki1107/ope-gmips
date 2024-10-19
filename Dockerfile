FROM python:3.9-slim-buster

WORKDIR /app
RUN apt-get update

RUN pip install --upgrade poetry \
  && poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock /app/

ENV PYTHONPATH=/app

RUN poetry install

ENTRYPOINT ["poetry", "run", "python"]
