FROM python:slim-bullseye

WORKDIR /app

COPY . . 

RUN pip install -r requirements.txt

CMD ["python3", "main.py"]