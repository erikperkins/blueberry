FROM python:2.7.12

RUN apt-get update -qq && apt-get install -y build-essential

ENV FLASK_HOME /flask_app
ENV FLASK_APP main.py
ENV FLASK_DEBUG 1

RUN mkdir $FLASK_HOME
WORKDIR $FLASK_HOME

ADD requirements.txt $FLASK_HOME
RUN pip install -r requirements.txt

ADD . $FLASK_HOME
