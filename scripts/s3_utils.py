import boto3
import os
from uuid import uuid4
from botocore.exceptions import ClientError

def get_s3_client():
    """Initialize S3 client from Streamlit secrets or env vars"""
    aws_access_key = None
    aws_secret_key = None
    aws_region = "us-east-1"
    
    try:
        import streamlit as st
        if hasattr(st, 'secrets'):
            aws_access_key = st.secrets.get("AWS_ACCESS_KEY_ID")
            aws_secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY")
            aws_region = st.secrets.get("AWS_REGION", "us-east-1")
    except Exception:
        pass
    
    if not aws_access_key:
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    if not aws_secret_key:
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not aws_region:
        aws_region = os.getenv("AWS_REGION", "us-east-1")
    
    if not aws_access_key or not aws_secret_key:
        raise RuntimeError(
            "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY "
            "in .streamlit/secrets.toml or as environment variables."
        )
    
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

def upload_to_s3_direct(file_bytes, bucket_name, s3_key):
    """Direct upload to S3 (can accept bytes or file path)"""
    s3 = get_s3_client()
    try:
        s3.put_object(Bucket=bucket_name, Key=s3_key, Body=file_bytes)
        return s3_key
    except ClientError as e:
        raise RuntimeError(f"Failed to upload to S3: {e}")

def download_from_s3(bucket_name, s3_key, local_path):
    """Download S3 object to local path"""
    s3 = get_s3_client()
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        return local_path
    except ClientError as e:
        raise RuntimeError(f"Failed to download from S3: {e}")

def delete_from_s3(bucket_name, s3_key):
    """Delete S3 object"""
    s3 = get_s3_client()
    try:
        s3.delete_object(Bucket=bucket_name, Key=s3_key)
    except ClientError as e:
        raise RuntimeError(f"Failed to delete from S3: {e}")

def list_s3_files(bucket_name, prefix):
    """List all files in S3 with given prefix"""
    s3 = get_s3_client()
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' not in response:
            return []
        return [obj['Key'] for obj in response['Contents']]
    except ClientError as e:
        raise RuntimeError(f"Failed to list S3 files: {e}")

def delete_s3_folder(bucket_name, folder_prefix):
    """Delete all files in an S3 'folder' (prefix)"""
    s3 = get_s3_client()
    try:
        # List all objects with the prefix
        objects = list_s3_files(bucket_name, folder_prefix)
        
        if not objects:
            return
        
        # Delete in batches of 1000 (S3 limit)
        for i in range(0, len(objects), 1000):
            batch = objects[i:i+1000]
            delete_dict = {'Objects': [{'Key': k} for k in batch]}
            s3.delete_objects(Bucket=bucket_name, Delete=delete_dict)
    except ClientError as e:
        raise RuntimeError(f"Failed to delete S3 folder: {e}")

def check_s3_object_exists(bucket_name, s3_key):
    """Check if S3 object exists"""
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=bucket_name, Key=s3_key)
        return True
    except ClientError:
        return False