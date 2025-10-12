from django.shortcuts import render, redirect
from django.contrib.auth import get_user_model, authenticate, login as auth_login, logout
from django.contrib.auth.decorators import login_required

from webnotice.utils import send_email

User = get_user_model()


def register(request):
    if request.user.is_authenticated:
        return redirect('/dashboard/')

    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        phone = request.POST.get("phone")
        user_type = request.POST.get("user_type")  # admin/user
        password1 = request.POST.get("password")
        image = request.FILES.get("image")

        # Check for existing user/email/phone
        if User.objects.filter(username=username).exists():
            return render(request, 'register.html', {"error": "Username already exists"})
        if User.objects.filter(email=email).exists():
            return render(request, 'register.html', {"error": "Email already exists"})
        if User.objects.filter(phone=phone).exists():
            return render(request, 'register.html', {"error": "Phone number already exists"})

        # Create user (password hashed automatically)
        user = User.objects.create_user(
            username=username,
            email=email,
            phone=phone,
            user_type=user_type,
            password=password1,
            image=image
        )

        auth_login(request, user)
        send_email(email, "Welcome to Web Notice", "Thank you for registering!")
        # Log the user in
        return redirect('/dashboard/')

    return render(request, "register.html")


def login(request):
    if request.user.is_authenticated:
        return redirect('/dashboard/')

    if request.method == "POST":
        username = request.POST.get('username')
        password = request.POST.get('password')
        print(username)
        print(password)
        user = authenticate(request, username=username, password=password)
        print(user)
        if user is not None:
            auth_login(request, user)
            return redirect('/dashboard/')
        else:
            return render(request, 'login.html', {"error": "Invalid email or password"})

    return render(request, 'login.html')


@login_required
def logout_view(request):
    logout(request)
    return redirect('login')


@login_required
def profile(request):
    user = request.user

    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        phone = request.POST.get("phone")
        user_type = request.POST.get("user_type")
        image = request.FILES.get("image")

        # Update user details
        user.username = username
        user.email = email
        user.phone = phone
        user.user_type = user_type
        if image:
            user.image = image
        user.save()

        return render(request, "profile.html", {"user": user, "message": "Profile updated successfully!"})

    return render(request, "profile.html", {"user": user})
