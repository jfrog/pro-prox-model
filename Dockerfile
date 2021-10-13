FROM python:3.8
ADD requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-dependencies -r requirements.txt