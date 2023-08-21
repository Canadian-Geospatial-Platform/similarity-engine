
try:
  import unzip_requirements
except ImportError:
  pass

import boto3
import logging 
from botocore.exceptions import ClientError
import pandas as pd 
import numpy as np
import json
import datetime
import io
import os 

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from gensim import matutils
from dynamodb import * 


#dev setting 
file_name = "Processed_records.parquet"
bucket_name_nlp = "nlp-data-preprocessing"
file_name_origianl = "records.parquet"
bucket_name = "webpresence-geocore-geojson-to-parquet-dev"

def lambda_handler(event, context):
    """
    #Change directory to /tmp folder, this is required if new files are created for lambda 
    os.chdir('/tmp')    #This is important
    #Make a directory
    if not os.path.exists(os.path.join('mydir')):
        os.makedirs('mydir')
    """    
    # Read the preprocessed data from S3 
    try:
        df_en = open_S3_file_as_df(bucket_name_nlp, file_name)
    except ClientError as e:
        print(e.response['Error']['Message'])
    
    # Use all data to train the model
    df = df_en[['features_properties_id', 'features_properties_title_en', 'features_properties_title_fr','metadata_en_processed']]
    print(f'The shape of the preprocessed df is {df.shape}')
    
    # Prepare the input for the Word2Vec model
    sentences = df['metadata_en_processed'].apply(lambda x: x.split(' ')).tolist()
    
    # Train the Word2Vec model
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    # Precompute L2-normalized vectors for better performance
    model.init_sims(replace=True)
    
    # Convert each sentence in 'metadata_preprocessed' into a vector
    vectors = df['metadata_en_processed'].apply(sentence_to_vector, model=model)
    # Replace the missing value in the 'features_properties_title_en' column with an empty string
    df['features_properties_title_en'].fillna('', inplace=True)
    
    # Calculate similarity between each vector and all others
    similarity_matrix = cosine_similarity(np.array(vectors.tolist()))
    
    
    # Upload the similar results as a AWS dynamodb 
    """
    The parquet lambda function has been modified to merge the similairy table with records.parquet everytime when records.parquet is updated. 
    
    """
    df['similarity'] = np.nan  # Initialize the column
    # For each text, find the top 10 most similar texts and save them as a JSON array object in the 'similarity' column
    df.reset_index(drop=True, inplace=True)
    for i in range(similarity_matrix.shape[0]):
        top_10_similar = np.argsort(-similarity_matrix[i, :])[1:11]  # Exclude the text itself
        sim_array = []
        for j, idx in enumerate(top_10_similar):
            sim_obj = {
            'sim': f'sim{j+1}',
            'features_properties_id': df.loc[idx, 'features_properties_id'],
            'features_properties_title_en': df.loc[idx, 'features_properties_title_en'],
            'features_properties_title_fr': df.loc[idx, 'features_properties_title_fr']
            }
            sim_array.append(sim_obj)
        df.loc[i, 'similarity'] = json.dumps(sim_array)
        
    df = df[['features_properties_id', 'similarity']]
    print(print(f'The shape of df after cosine similarity analysis is {df.shape}'))
    
    # Create a dynamoDB table 'similarity, if similarity is exist, delete the old one first 
    # Check if the table already exists
    dynamodb = boto3.resource('dynamodb')
    existing_tables = dynamodb.meta.client.list_tables()['TableNames']
    if 'similarity' in existing_tables:
        print("Table 'similarity' already exists. Deleting the similarity table is required")
        try: 
            """
            Gets a waiter object that waits for a DynamoDB table to be deleted. 
            This waiter will poll the DynamoDB service to check if the table still exists, and will stop once the table is deleted.
            """
            client = boto3.client('dynamodb')
            delete_table(TableName='similarity')
            waiter = client.get_waiter('table_not_exists')
            waiter.wait(TableName='similarity')
        except ClientError as e:
            print(e)
    #Create table         
    try: 
        client = boto3.client('dynamodb')
        create_table_similarity(TableName='similarity', dynamodb=None)
        waiter = client.get_waiter('table_exists')
        waiter.wait(TableName='similarity')
    except ClientError as e:
            print(e)  
            
    """DEBUG
    #Check if empty string in the primary key before scan the table 
    empty_string_rows = df[df['features_properties_id'] == '']
    print(f'Number of NA values in the df id column is \n {empty_string_rows}')
    """
    #Remove rows with empty string in 'features_properties_id', primary key can not be empty in DynamoDB table 
    df_cleaned = df[df['features_properties_id']!='']
    rows_removed = df.shape[0] - df_cleaned.shape[0]
    print(f'Removed {rows_removed} rows with empyt string in features_properties_id')
    #Batch write to table 
    batch_write_items_into_table(df_cleaned, TableName='similarity')
    
    
    
# Function to read the parquet file as pandas dataframe 
def open_S3_file_as_df(bucket_name, file_name):
    """Open a S3 parquet file from bucket and filename and return the parquet as pandas dataframe
    :param bucket_name: Bucket name
    :param file_name: Specific file name to open
    :return: body of the file as a string
    """
    try: 
        s3 = boto3.resource('s3')
        object = s3.Object(bucket_name, file_name)
        body = object.get()['Body'].read()
        df = pd.read_parquet(io.BytesIO(body))
        print(f'Loading {file_name} from {bucket_name} to pandas dataframe')
        return df
    except ClientError as e:
        logging.error(e)
        return e
        
# Upload the duplicate date to S3 as a parquet file 
def upload_dataframe_to_s3_as_parquet(df, bucket_name, file_key):
    # Save DataFrame as a Parquet file locally
    parquet_file_path = 'temp.parquet'
    df.to_parquet(parquet_file_path, index=False)  # Set index to False

    # Create an S3 client
    s3_client = boto3.client('s3')

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
    
# Function to computing the vector representations of the texts, from sentence to vector 
def sentence_to_vector(sentence, model):
    words = str(sentence).split()
    vector = np.mean([model.wv[word] for word in words if word in model.wv.key_to_index], axis=0)
    return vector if isinstance(vector, np.ndarray) else np.zeros(model.vector_size)