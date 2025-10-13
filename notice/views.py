from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required

from notice.models import Notice
from webnotice.utils import send_email

from django.contrib.auth import get_user_model
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

from django.db.models import Q

from django.db.models import Q

@login_required
def notices(request):
    query = request.GET.get('q', '')
    priority = request.GET.get('priority', '')

    notices = Notice.objects.filter(is_active=True)

    if query:
        notices = notices.filter(Q(title__icontains=query) | Q(content__icontains=query))

    if priority:
        notices = notices.filter(priority=priority)

    notices = notices.order_by('-created_at')

    context = {'notices': notices, 'query': query, 'priority': priority}

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
        attachment = request.FILES.get('attachment')  # file

        # Create and save notice
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
        users = User.objects.filter(is_active=True)  # fetch active users
        subject = f"New Notice: {notice.title}"
        body = f"""
Hello,

A new notice has been posted by {request.user.get_full_name() or request.user.username}.

Title: {notice.title}
Content: {notice.content}
Priority: {notice.priority}
"""
        for user in users:
            if user.email:  # make sure user has an email
                send_email(to=user.email, subject=subject, body=body)

        return redirect('view_notices')  # redirect to notice list

    return render(request, 'create.html')