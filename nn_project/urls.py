from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),          # the built-in admin panel
    path("", include("object_detection.urls")),  # hand the homepage to our app
]

# While DEBUG is on, let Django serve the uploaded images.
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)