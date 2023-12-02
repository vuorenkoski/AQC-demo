# Community detection demo

Framework: Python/Django/Bootstrap/Gunicorn


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
