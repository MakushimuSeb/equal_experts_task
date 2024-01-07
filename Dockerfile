FROM python:3.8-slim

WORKDIR /usr/src/app

RUN mkdir models

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY api.py ./
COPY config.yaml ./

EXPOSE 5000

CMD ["python", "api.py"]
