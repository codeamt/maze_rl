FROM python:3.7

MAINTAINER "annmargaret.tutu@icloud.com"

RUN apt-get update -y && \
 apt-get install -y python-pip python-dev

WORKDIR /app

COPY ./requirements.txt /

RUN pip install -r requirements.txt

COPY ./code /

ENTRYPOINT ["python", "main.py"]