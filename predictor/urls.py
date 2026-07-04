from django.urls import path

from . import views


urlpatterns = [

    path(
        "",
        views.welcome,
        name="welcome",
    ),

    path(
        "predict/",
        views.home,
        name="home",
    ),

    path(
        "history/",
        views.history,
        name="history",
    ),

]