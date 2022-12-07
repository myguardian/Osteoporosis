FROM tiangolo/uvicorn-gunicorn:python3.8

WORKDIR ./app

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip\
    && pip install --no-cache-dir -r requirements.txt

COPY ./app .

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]