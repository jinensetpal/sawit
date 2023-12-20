FROM python:3.11

ADD requirements.txt /

RUN pip install -r /requirements.txt

ADD arxiv_data_collection.py /

ADD config.ini /

ENV PYTHONBUFFERED=1

CMD [ "python", "./arxiv_data_collection.py" ]
