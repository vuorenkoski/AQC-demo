FROM python:3.9.18-slim-bullseye
WORKDIR /user/src/app
RUN pip install Django dwave-ocean-sdk matplotlib
COPY . /user/src/app/
CMD python manage.py runserver 0.0.0.0:8000