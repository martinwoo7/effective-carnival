FROM python:3.10-slim

WORKDIR /app
ADD ./backend/requirements.txt /app/backend/

# RUN apk --no-cache add musl-dev linux-headers g++

RUN pip install --upgrade pip
RUN pip install gunicorn
RUN pip install -r backend/requirements.txt

ADD ./docker /app/docker
ADD ./backend /app/backend