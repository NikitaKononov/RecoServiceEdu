FROM python:3.8-buster as build

COPY . .

RUN pip install -U --no-cache-dir pip poetry setuptools wheel && \
    poetry build -f wheel && \
    poetry export -f requirements.txt -o requirements.txt --without-hashes && \
    pip wheel -w dist -r requirements.txt

# setup timezone
ENV TZ=UTC
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN pip install -U --no-cache-dir pip dist/*.whl && \
    rm -rf dist
RUN apt-get update && apt-get install build-essential -y
RUN pip install python-multipart joblib hnswlib rectools

CMD ["gunicorn", "main:app", "-c", "gunicorn.config.py"]
