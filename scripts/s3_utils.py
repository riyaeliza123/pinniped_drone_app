import boto3
import os
from uuid import uuid4
from botocore.exceptions import ClientError

def get_s3_client():
    """Initialize S3 client from Streamlit secrets or env vars"""
    try:
        import streamlit as st
        aws_access_key = st.secrets.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = st.secrets.get("AWS_SECRET_ACCESS_KEY")
        aws_region = st.secrets.get("AWS_REGION", "us-east-1")
    except:
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_region = os.getenv("AWS_REGION", "us-east-1")
    
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )

def generate_presigned_url(bucket_name, s3_key, expiration=3600):
    """Generate presigned GET URL for Roboflow to access the image"""
    s3 = get_s3_client()
    try:
        url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': s3_key},
            ExpiresIn=expiration
        )
        return url
    except ClientError as e:
        raise RuntimeError(f"Failed to generate presigned URL: {e}")

def upload_to_s3_direct(file_bytes, bucket_name, original_filename):
    """Direct upload to S3"""
    s3 = get_s3_client()
    unique_key = f"uploads/{uuid4().hex}_{original_filename}"
    s3.put_object(Bucket=bucket_name, Key=unique_key, Body=file_bytes)
    return unique_key

def download_from_s3(bucket_name, s3_key, local_path):
    """Download S3 object to local path"""
    s3 = get_s3_client()
    s3.download_file(bucket_name, s3_key, local_path)
    return local_path

def delete_from_s3(bucket_name, s3_key):
    """Delete S3 object"""
    s3 = get_s3_client()
    s3.delete_object(Bucket=bucket_name, Key=s3_key)

def check_s3_object_exists(bucket_name, s3_key):
    """Check if S3 object exists"""
    s3 = get_s3_client()
    try:
        s3.head_object(Bucket=bucket_name, Key=s3_key)
        return True
    except ClientError:
        return False