FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

COPY . .

RUN mkdir -p /app/logs && chmod 777 /app/logs

CMD ["python", "run.py"]