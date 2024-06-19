FROM python:3.10-buster

ADD . /opt/ml_in_app
WORKDIR /opt/ml_in_app

# install packages by conda
RUN pip install -r requirements_prod.txt

#RUN pip install numpy==1.26.3
CMD ["python", "app.py"]
