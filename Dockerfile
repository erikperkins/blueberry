FROM python:3.7.2-stretch

RUN apt-get update -qq && apt-get install -y build-essential

ENV FLASK_HOME /blueberry
RUN mkdir $FLASK_HOME
WORKDIR $FLASK_HOME

ADD requirements.txt $FLASK_HOME
RUN pip install -r requirements.txt

ADD . $FLASK_HOME
