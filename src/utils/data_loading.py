# Load data from aws.
import boto3
import logging 
from botocore.exceptions import ClientError
import pandas as pd 
import io
import os 
import json

def open_S3_file_as_df(bucket_name, file_name, session):
    """Open a S3 parquet file from bucket and filename and return the parquet as pandas dataframe
    :param bucket_name: Bucket name
    :param file_name: Specific file name to open
    :return: body of the file as a string
    """
    with open('aws_credentials.json') as f:
        secrets = json.load(f)
    

    try: 
        s3 = session.resource('s3')
        object = s3.Object(bucket_name, file_name)
        body = object.get()['Body'].read()
        df = pd.read_parquet(io.BytesIO(body))
        print(f'Loading {file_name} from {bucket_name} to pandas dataframe')
        return df
    except ClientError as e:
        logging.error(e)
        return e
    
    
def preprocess_aws_parquet_data(df_en):
    # df_en_deduplicated = df_en.drop_duplicates(subset='metadata_en_processed')
    df = df_en[['features_properties_id', 'features_properties_title_en', 'metadata_en_processed']]
    return df
    
    
def load_processed_parquet(from_local=False):
    # Instantiate the boto3 client.
    if from_local:
        df_en = pd.read_parquet('/home/rsaha/projects/similarity-engine/data/Processed_records_sep_15.parquet')
        df_en = preprocess_aws_parquet_data(df_en)
        return df_en
    with open('aws_credentials.json') as f:
        secrets = json.load(f)
    
    # We might want to remove this later on.
    session = boto3.Session(
        aws_access_key_id=secrets['ACCESS_KEY'],
        aws_secret_access_key=secrets['SECRET_KEY'],
        aws_session_token=secrets['SESSION_TOKEN']
    )
    bucket_name_nlp = "nlp-data-preprocessing"
    file_name = "Processed_records.parquet"
    df_en = open_S3_file_as_df(bucket_name_nlp, file_name, session)
    df_en = preprocess_aws_parquet_data(df_en)
    return df_en


def upload_dataframe_to_s3_as_parquet(df, bucket_name, file_key):
    parquet_file_path = 'temp.parquet'
    df.to_parquet(parquet_file_path, index=False)  # Set index to False

    # Create an S3 client
    with open('aws_credentials.json') as f:
        secrets = json.load(f)
    s3_client = boto3.client('s3',
                            aws_access_key_id=secrets['ACCESS_KEY'],
                            aws_secret_access_key=secrets['SECRET_KEY'],
                            aws_session_token=secrets['SESSION_TOKEN']
                            )

    # Upload the Parquet file to S3 bucket
    try:
        response = s3_client.upload_file(parquet_file_path, bucket_name, file_key)
        os.remove(parquet_file_path)
        print(f'Uploading {file_key} to {bucket_name} as parquet file')
        # Delete the local Parquet file
        return True
    except ClientError as e:
        logging.error(e)
        return False
    
# Load data and print it.
# df_en = load_processed_parquet(from_local=True)

# print(f'The shape of the raw metadata parquet dataset is {df_en.shape}')
