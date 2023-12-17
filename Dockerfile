FROM python:3.9.18-slim-bullseye
WORKDIR /user/src/app
RUN pip install django dwave-ocean-sdk gunicorn
COPY . /user/src/app/
CMD gunicorn -b :8000 qcdemo.wsgi