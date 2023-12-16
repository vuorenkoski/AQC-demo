# AQC demo

This demo run some graph algorithms with quantum annealer (D-Wave).

Framework: Python/Django/Bootstrap/Gunicorn/D3 data visualization


## Run locally

```
pip install Django dwave-ocean-sdk matplotlib 
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
