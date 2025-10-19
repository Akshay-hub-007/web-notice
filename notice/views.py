from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.core.mail import send_mass_mail
from notice.models import Notice
from webnotice.utils import send_email
import cloudinary
from django.contrib.auth import get_user_model
from django.db.models import Q
import datetime
User = get_user_model()

# Create your views here.

@login_required
def dashboard(request):
    user = request.user

    # Notice stats
    total_notices = Notice.objects.count()
    active_notices = Notice.objects.filter(is_active=True).count()
    urgent_notices = Notice.objects.filter(priority='urgent').count()
    important_notices = Notice.objects.filter(priority='important').count()
    normal_notices = Notice.objects.filter(priority='normal').count()

    # User stats
    total_users = User.objects.count()
    admin_users = User.objects.filter(user_type='admin').count()
    normal_users = User.objects.filter(user_type='user').count()

    context = {
        'user': user,
        'total_notices': total_notices,
        'active_notices': active_notices,
        'urgent_notices': urgent_notices,
        'important_notices': important_notices,
        'normal_notices': normal_notices,
        'total_users': total_users,
        'admin_users': admin_users,
        'normal_users': normal_users,
    }
    return render(request, 'dashboard.html', context)

from django.contrib.auth.decorators import login_required
from django.db.models import Q
from datetime import datetime

@login_required
def notices(request):
    query = request.GET.get('q', '')
    priority = request.GET.get('priority', '')
    start_date = request.GET.get('start_date', '')
    end_date = request.GET.get('end_date', '')

    notices = Notice.objects.filter(is_active=True)

    # search filter
    if query:
        notices = notices.filter(Q(title__icontains=query) | Q(content__icontains=query))

    # priority filter
    if priority:
        notices = notices.filter(priority=priority)

    # date range filter
    if start_date or end_date:
        if start_date:
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if end_date:
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        if start_date and end_date:
            notices = notices.filter(
                Q(created_at__date__gte=start_date) & Q(expiry_date__date__lte=end_date)
            )
        elif start_date:
            notices = notices.filter(created_at__date__gte=start_date)
        elif end_date:
            notices = notices.filter(expiry_date__date__lte=end_date)

    notices = notices.order_by('-created_at')

    context = {'notices': notices, 'query': query, 'priority': priority}

    # support AJAX reload
    if request.headers.get('x-requested-with') == 'XMLHttpRequest':
        return render(request, 'notices_list.html', context)

    return render(request, 'notices.html', context)


@login_required
def manage(request):
    return render(request,'manage.html')


def home(request):
    return render(request, "home.html")

@login_required
def create(request):
    if request.method == 'POST':
        title = request.POST.get('title')
        content = request.POST.get('content')
        priority = request.POST.get('priority', 'normal')
        expiry_date = request.POST.get('expiry_date')  # optional
        is_active = request.POST.get('is_active') == 'on'  # checkbox
        attachment = request.FILES.get('attachment')
        print(attachment.__dict__)
        notice = Notice(
            title=title,
            content=content,
            priority=priority,
            expiry_date=expiry_date if expiry_date else None,
            is_active=is_active,
            attachment=attachment,
            posted_by=request.user
        )
        notice.save()

        # Send email to all users
        users = User.objects.exclude(email = '')  # fetch active users
        subject = f"New Notice: {notice.title}"
        body = f"""
Hello,

A new notice has been posted by {request.user.get_full_name() or request.user.username}.

Title: {notice.title}
Content: {notice.content}
Priority: {notice.priority}
"""

        emails = [ (subject, body, None,[user.email]) for user in users]
        send_mass_mail(emails ,fail_silently=False)
        return redirect('view_notices')  # redirect to notice list

    return render(request, 'create.html')