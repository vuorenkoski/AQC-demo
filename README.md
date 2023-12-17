# AQC demo

This demo run some graph algorithms with quantum annealer (D-Wave).

Framework: Python/Django/Gunicorn/Bootstrap/D3.js

## Run locally

```
pip install django dwave-ocean-sdk
python manage.py runserver
```

## Docker

### Build

```
sudo docker build -t aqc_demo .
```

### Run

```
sudo docker run -p 8000:8000 aqc_demo
```

### Dockerhub

https://hub.docker.com/r/vuorenkoski/aqc_demo


```
docker pull vuorenkoski/aqc_demo
```
