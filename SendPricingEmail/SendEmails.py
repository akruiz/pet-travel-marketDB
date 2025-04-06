import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart

# New gmail created with 2FA: spat.gt0425@gmail.com
# App password:   sywg ndvs lzew ecet
sender_email = 'spat.gt0425@gmail.com'
sender_pw = 'nrpd agse cdny vioh'

# Recipients
df_excel = pd.read_excel("Emails Test.xlsx", engine="openpyxl")
df = df_excel[['Company', 'CityState', 'Email', 'PrimaryAirport', 'SecondaryAirport']]
df.loc[:, 'Email'] = df['Email'].fillna('')

def send_mail(recipient_email, subject, message, server='smtp.gmail.com',
              from_email=sender_email, password=sender_pw):
    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = ', '.join(recipient_email)
    msg.set_content(message)

    # Set up server connection using Gmail sender --
    # https://stackoverflow.com/questions/10147455/how-to-send-an-email-with-gmail-as-provider-using-python
    with smtplib.SMTP(server, 587) as server_connection:
        server_connection.ehlo()
        server_connection.starttls()
        server_connection.login(from_email, password)
        server_connection.send_message(msg)

    print(f"Email sent to: {recipient_email}")

subject = "Quote for Shipping Shepherd to London, UK"
body_message = """\
Hello,

I will be relocating to London, UK in September and would like to bring my 3 years old Rocky the German Shepherd with me. Given his weight of ~90 lbs., and 30 inches height, I understand that he would need to travel via cargo. He is fully vaccinated and has a microchip, and I’ve reviewed the health certificate requirements. Could you please provide me with a quote for shipping him from {airport} to London? I would like to drop him off and pick him up at the airports, and I would prefer to be on the same flight if possible, as I have not purchased my ticket yet. 

In addition to the quote, I would appreciate it if you could provide the following details:

    Estimated duration of the shipment if he can't flight with me.
    Flight options if I want to be in the same flight with him.
    Whether there is an option for insurance and what it covers and its cost.
    The total cost and what it includes, any additional fees or charges that may apply.
    
    
Thank you in advance,
Tran
"""



for index, row in df.iterrows():
    recipient_email = row['Email']
    airport = row['PrimaryAirport']
    business = row['Company']

    if recipient_email and "@" in recipient_email:
        body = body_message.format(airport=airport)
        send_mail([recipient_email], subject, body,
          from_email=sender_email, password=sender_pw)
    else:
        print(f"Skipped sending email to: {business}")
