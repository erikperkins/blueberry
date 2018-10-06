FROM python:2.7.12

RUN apt-get update -qq && apt-get install -y build-essential

ENV BLUEBERRY_HOME /blueberry
ENV blueberry main.py
ENV BLUEBERRY_DEBUG 1

RUN mkdir $BLUEBERRY_HOME
WORKDIR $BLUEBERRY_HOME

ADD requirements.txt $BLUEBERRY_HOME
RUN pip install -r requirements.txt

ADD . $BLUEBERRY_HOME
