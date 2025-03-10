import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart

# New gmail created with 2FA: t.nguyen070196@gmail.com
# App password:   sywg ndvs lzew ecet
sender_email = 't.nguyen070196@gmail.com'
sender_pw = 'sywg ndvs lzew ecet'

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

subject = "Quote for Shipping ~45 lbs., Golden Dooble to London, UK"
body_message = """\
Hello,

I will be relocating to London, UK in the next two months and would like to bring my 3 years old Rocky the Golden Doodle with me. Given his weight of 45 lbs, I understand that he would need to travel via cargo. He is fully vaccinated and has a microchip, and I’ve reviewed the health certificate requirements. Could you please provide me with a quote for shipping him from {airport} to London?

In addition to the quote, I would appreciate it if you could provide the following details:

    Estimated duration of the shipment
    Breakdown of the total cost and what it includes
    Whether there is an option for insurance and what it covers
    Any additional fees or charges that may apply
    
Thank you,

"""

for index, row in df.iterrows():
    recipient_email = row['Email']
    airport = row['PrimaryAirport']

    body = body_message.format(airport=airport)

    send_mail([recipient_email], subject, body,
              from_email=sender_email, password=sender_pw)