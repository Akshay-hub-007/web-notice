import smtplib
from email.mime.text import MIMEText

def send_email(to, subject, message):
    """Reusable function for sending emails via SMTP"""
    sender_email = "your_email@gmail.com"
    sender_password = "your_app_password"  # use Gmail app password

    try:
        msg = MIMEText(message, "plain")
        msg["From"] = sender_email
        msg["To"] = to
        msg["Subject"] = subject

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        print(f"✅ Email sent successfully to {to}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
