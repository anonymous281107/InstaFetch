from django.urls import path

from . import views

urlpatterns = [
    path('filterData',views.api_home),
    path('upload',views.insertToCone),
    path('getMenu',views.getFullMenu),
    path('uploadToChroma',views.uploadToChroma),
    path('checkCollectionDetails',views.checkCollectionDetails),
]