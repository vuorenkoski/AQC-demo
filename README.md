# AQC demo

This demo run some graph algorithms with quantum annealer (D-Wave).

Framework: Python / Django / Gunicorn / Bootstrap / D3.js

## Run locally

```
pip install django dwave-ocean-sdk
python manage.py runserver
```

## Docker

### Build

```
sudo docker build -t aqc_demo:v0.5 .
```

### Run

```
sudo docker run -p 8000:8000 aqc_demo:v0.5
```

After this command application can be accesses from address http://127.0.0.1:8000/

### Dockerhub

https://hub.docker.com/r/vuorenkoski/aqc_demo


```
sudo docker pull vuorenkoski/aqc_demo:v0.5
```
