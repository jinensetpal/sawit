#!/usr/bin/env python3

from dagshub.data_engine import datasources
from dagshub import get_repo_bucket_client
from src import const
import pandas as pd
import dagshub
import arxiv
import boto3
import re
import os

REPO_NAME = config.get('dagshub','repo_name')
USER_NAME = config.get('dagshub','user_name')
TOKEN = config.get('dagshub','token')
dagshub.auth.add_app_token(TOKEN)
datasource_name = config.get('dagshub','datasource_name')

# dagshub.init(REPO_NAME,USER_NAME)


s3_client = get_repo_bucket_client(f"{USER_NAME}/{REPO_NAME}", flavor="boto")

column_names = ['Path','Entry Id','Title','Summary','Primary Category','Category','PDF Link']

# topic = input("Enter the topic you need to search for : ")
topic = config.get('data','papers')
number_of_papers = config.get('data','number_of_papers')
search = arxiv.Search(
  query = topic,
  max_results = 30,
  sort_by = arxiv.SortCriterion.Relevance,
  sort_order = arxiv.SortOrder.Descending
)
all_data = []
os.makedirs(topic, exist_ok = True)

for result in arxiv.Client().results(search):
  path = f"{result.title}.pdf"
  all_data.append([path,
                   result.entry_id,
                   result.title,
                   result.summary,
                   result.primary_category,
                   result.categories,
                   result.pdf_url])

  cleaned_title = re.sub(r'[^a-zA-Z0-9\s]', '', result.title)
  print("Downloading file ", cleaned_title)
  result.download_pdf(dirpath=f"./{topic}", filename=f"{cleaned_title}.pdf")
  s3_client.upload_file(f"./{topic}/{cleaned_title}.pdf", REPO_NAME, f"{topic}/{cleaned_title}.pdf")

metadata = pd.DataFrame(all_data, columns=column_names)
ds = datasources.get_or_create(repo=f"{USER_NAME}/{REPO_NAME}", name=datasource_name, path="Science")
ds.upload_metadata_from_dataframe(metadata, path_column="Path")

print("Papers Uploaded to your repo")

