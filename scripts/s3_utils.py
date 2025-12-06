import boto3
from botocore.exceptions import ClientError
import streamlit as st

def get_s3_client():
    """Get S3 client with credentials from secrets or environment"""
    try:
        return boto3.client(
            's3',
            aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=st.secrets.get("AWS_SECRET_ACCESS_KEY"),
            region_name=st.secrets.get("AWS_DEFAULT_REGION", "us-west-2")
        )
    except:
        return boto3.client('s3')

def delete_s3_folder(bucket_name, folder_prefix):
    """Delete all objects in a specific S3 folder"""
    s3_client = get_s3_client()
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=folder_prefix)
        
        delete_objects = []
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    delete_objects.append({'Key': obj['Key']})
        
        if delete_objects:
            s3_client.delete_objects(
                Bucket=bucket_name,
                Delete={'Objects': delete_objects}
            )
        return True
    except ClientError as e:
        print(f"Error deleting folder {folder_prefix}: {e}")
        return False

def clean_entire_dropbox_extracts_folder(bucket_name):
    """Delete all content in dropbox_extracts/ folder"""
    return delete_s3_folder(bucket_name, "dropbox_extracts/")

def upload_to_s3_direct(file_data, bucket_name, s3_key):
    """Upload file data directly to S3"""
    s3_client = get_s3_client()
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=file_data
        )
        return True
    except ClientError as e:
        print(f"Error uploading to S3: {e}")
        return False

def download_from_s3(bucket_name, s3_key, local_path):
    """Download file from S3 to local path"""
    s3_client = get_s3_client()
    try:
        s3_client.download_file(bucket_name, s3_key, local_path)
        return True
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        return False

def list_s3_files(bucket_name, prefix):
    """List all files in S3 bucket with given prefix"""
    s3_client = get_s3_client()
    files = []
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    if not obj['Key'].endswith('/'):  # Skip folder markers
                        files.append(obj['Key'])
        return files
    except ClientError as e:
        print(f"Error listing S3 files: {e}")
        return []