import boto3

bucket = "streamflow-app-data"

s3 = boto3.client("s3")          # uses your configured credentials/region
paginator = s3.get_paginator("list_objects_v2")

for page in paginator.paginate(Bucket=bucket):
    for obj in page.get("Contents", []):   # empty list if page has no objects
        print(obj["Key"])