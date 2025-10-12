
# Register your models here.
from django.contrib import admin
from .models import Notice

@admin.register(Notice)
class NoticeAdmin(admin.ModelAdmin):
    list_display = ('title', 'priority', 'posted_by', 'created_at')
    list_filter = ('priority', 'created_at')
    search_fields = ('title', 'content')
