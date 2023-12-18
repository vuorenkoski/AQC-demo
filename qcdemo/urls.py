from django.urls import path
from . import views
from .algorithms import apsp, gi, cd
from django.contrib.staticfiles.urls import staticfiles_urlpatterns

urlpatterns = [
    path('cd/', cd.index),
    path('gi/', gi.index),
    path('apsp/', apsp.index),
    path('', views.index),
    path('docs/', views.docs),
]

urlpatterns += staticfiles_urlpatterns()
