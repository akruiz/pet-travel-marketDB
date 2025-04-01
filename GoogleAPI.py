import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery

# Service account key file
key_file = 'spat-455404-49dc6376255b.json'
scopes = ["https://www.googleapis.com/auth/cloud-platform",
          "https://www.googleapis.com/auth/drive"]
credentials = service_account.Credentials.from_service_account_file(key_file, scopes=scopes)

client = bigquery.Client(credentials=credentials, project="spat-455404")
project_id = "spat-455404"
dataset_id = "SPAT"
tables = ["BBB", "GoogleReview", "IPATA", "Map_Pricing"]

# Fetch data
df_dict = {}
for table in tables:
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table}`"
    df = client.query(query).to_dataframe()
    df_dict[table] = df.iloc[1:]  # Skip the first row because it's duplicated in the Google Sheet

BBB_df = df_dict.get("BBB", pd.DataFrame())
GoogleReview_df = df_dict.get("GoogleReview", pd.DataFrame())
IPATA_df = df_dict.get("IPATA", pd.DataFrame())
MapPricing_df = df_dict.get("MapPricing", pd.DataFrame())

