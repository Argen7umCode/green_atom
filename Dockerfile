FROM python:slim-bullseye

WORKDIR /app

COPY . . 

RUN python -m pip install -r requirements.txt

CMD ["python" , "main.py"]