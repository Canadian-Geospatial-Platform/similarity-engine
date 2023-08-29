import boto3
import logging 
from botocore.exceptions import ClientError
import pandas as pd 
import io
import os 

import nltk
from nltk.corpus import stopwords          # module for stop words that come with NLTK
from nltk.stem import PorterStemmer        # module for stemming
from nltk.tokenize import word_tokenize   # module for tokenizing strings  

#nltk.download('punkt')
#nltk.download('stopwords')

import re

# environment variables for lambda
file_name = os.environ['FILE_NAME']
bucket_name = os.environ['BUCKET_NAME']
bucket_name_nlp = os.environ['BUCKET_NAME_NLP']

"""
#dev setting  -- comment out for release
file_name = "records.parquet"
bucket_name = "webpresence-geocore-geojson-to-parquet-dev"
bucket_name_nlp='nlp-data-preprocessing'
"""

def lambda_handler(event, context):
    
    #Change directory to /tmp folder
    os.chdir('/tmp')    #This is important
    """
    #Make a directory
    if not os.path.exists(os.path.join('mydir')):
        os.makedirs('mydir')
    """  
    df = open_S3_file_as_df(bucket_name, file_name)
    print(f'The shape of the raw metadata parquet dataset is {df.shape}')

    # Select key columns, currently only english
    df_en = df[['features_properties_id', 'features_properties_title_en','features_properties_title_fr','features_properties_description_en','features_properties_keywords_en']]
    # Replace NaN and "Not Available; Indisponible" with empty string 
    print("The NaN values in the English columns are \n") 
    df_en = df_en.fillna('')
    df_en = df_en.replace('Not Available; Indisponible', '')

    # Cancadenate the selected variables to a new variable
    df_en['metadata_en'] = df_en['features_properties_title_en'] + ' ' + df_en['features_properties_description_en'] + ' ' + df_en['features_properties_keywords_en'] 
    if df_en['metadata_en'].isnull().any():
        df_en['metadata_en'] = df_en['metadata_en'].fillna('')

    # Delete duplications in the metadata_en and uuid 
    duplicateRowsDF1 = df_en[df_en.duplicated(['features_properties_id'], keep=False)]
    df_en = df_en.drop_duplicates(subset=['features_properties_id'], keep='first')
    # Find rows with duplications in 'metadata_en'
    duplicateRowsDF2 = df_en[df_en.duplicated(['metadata_en'], keep=False)]
    df_en = df_en.drop_duplicates(subset=['metadata_en'], keep='first')
    
    
    # Upload the duplicate date to S3 as a csv file
    duplicateRowsDF = pd.concat([duplicateRowsDF1, duplicateRowsDF2])
    print('The length of the uuids of the duplicated rows are: ')
    #print(len(duplicateRowsDF['features_properties_id'].unique()))
    #print(duplicateRowsDF['features_properties_id'].unique())
   
    # Save to temp  folder, see https://iotespresso.com/temporary-storage-during-aws-lambda-runtime-python/
    save_path = os.path.join(os.getcwd(), 'duplicateRowsDF')
    duplicateRowsDF.to_csv(save_path)
    df_fetched= pd.read_csv(save_path)
    
    upload_dataframe_to_s3_as_parquet(df=df_fetched,  bucket_name=bucket_name_nlp, file_key='Duplicated_records.parquet')  
    
    # Apply text preprocessing to the 'Text' column
    df_en['metadata_en_processed'] = df_en['metadata_en'].apply(process_tokens)

    # Upload the processed dataframe to S3
    upload_dataframe_to_s3_as_parquet(df=df_en,  bucket_name=bucket_name_nlp, file_key='Processed_records.parquet') 
    
    
    
# Function to open a S3 file from bucket and filename and return the parquet as pandas dataframe
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
    df.to_parquet(parquet_file_path)

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
    
#Preprocessing 
# Tokenize the string into words
def tokenize_text(text):
    return(word_tokenize(text.lower()))

# remove stop words and punctuation for string
def remove_stopwords_punctuation_tokens(tokens):
    stop_words = stopwords.words('english') 
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    return tokens

def remove_apostrophe_tokens(tokens):
    tokens = [re.sub(r"\'s$", "", token) for token in tokens]
    return tokens

def stemming_tokens(tokens):
    # Instantiate stemming class
    stemmer = PorterStemmer() 
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def process_tokens(tokens):
    # Tokenize text and lower case
    tokens = tokenize_text(tokens)
    # Remove stop words and punctuation
    tokens = remove_stopwords_punctuation_tokens(tokens)
    # Remove apostrophe
    tokens = remove_apostrophe_tokens(tokens)
    # Stemming
    tokens = stemming_tokens(tokens)
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text
