import smtplib
from email.message import EmailMessage

def send_email(to, subject, body, use_ssl=False, timeout=20):
    """Send email via Gmail SMTP (TLS or SSL)"""
    sender_email = "akshaykalangi54@gmail.com"
    sender_password = "hyss gpbd icpd ctjj"
    print(to)
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if use_ssl:
            # SSL on port 465
            server = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=timeout)
        else:
            # TLS on port 587
            server = smtplib.SMTP("smtp.gmail.com", 587, timeout=timeout)
            server.ehlo()
            server.starttls()
            server.ehlo()

        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print(f"Failed to send email: {type(e).__name__}: {e}")
