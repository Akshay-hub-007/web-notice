from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    USER_TYPE_CHOICES = (
        ('admin', 'Admin'),
        ('user', 'User'),
    )

    phone = models.CharField(max_length=15, unique=True)
    user_type = models.CharField(max_length=10, choices=USER_TYPE_CHOICES, default='user')
    image = models.URLField(max_length=500, blank=True, null=True)  # ✅ Store Cloudinary URL
    image_public_id = models.CharField(max_length=255, blank=True, null=True)  # ✅ Store Cloudinary public ID
