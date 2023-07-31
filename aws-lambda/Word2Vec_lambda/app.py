
import boto3
import logging 
from botocore.exceptions import ClientError
import pandas as pd 
import io
import os 

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from gensim import matutils
import numpy as np
from tqdm import tqdm


file_name = "Processed_records.parquet"
bucket_name_nlp = "nlp-data-preprocessing"
file_name_origianl = "records.parquet"
bucket_name = "webpresence-geocore-geojson-to-parquet-dev"

def lambda_handler(event, context):
    #Change directory to /tmp folder, this is required if new files are created for lambda 
    os.chdir('/tmp')    #This is important
    #Make a directory
    if not os.path.exists(os.path.join('mydir')):
        os.makedirs('mydir')
        
    # Read the preprocessed data from S3 
    df_en = open_S3_file_as_df(bucket_name_nlp, file_name)
    
    # Get a sample of 500 rows as the training data 
    df = df_en[['features_properties_id', 'features_properties_title_en', 'metadata_en_processed']]
    #df = df.sample(n=500, random_state=1)
    # Use all data to train the model
    df.head()
    print(df.shape)

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
    
    # Initialize new columns for the top 5 similar texts
    df['sim1'], df['sim2'], df['sim3'], df['sim4'], df['sim5'] = "", "", "", "", ""
    
    # For each text, find the top 5 most similar texts and append their 'features_properties_title_en' as new columns
    df.reset_index(drop=True, inplace=True)
    for i in tqdm(range(similarity_matrix.shape[0])):
        top_5_similar = np.argsort(-similarity_matrix[i, :])[1:6]  # Exclude the text itself
        df.loc[i, ['sim1', 'sim2', 'sim3', 'sim4', 'sim5']] = df.loc[top_5_similar, 'features_properties_title_en'].values

    # Read the original parquet file and merge by features_properties_id
    df_original = open_S3_file_as_df(bucket_name, file_name_origianl)
    merged_df = df_original.merge(df[['features_properties_id', 'sim1', 'sim2', 'sim3', 'sim4', 'sim5']], on='features_properties_id', how='left')
    
    # Save to temp  folder, see https://iotespresso.com/temporary-storage-during-aws-lambda-runtime-python/
    save_path = os.path.join(os.getcwd(), 'mydir', 'merged_df')
    merged_df.to_csv(save_path)
    df_fetched= pd.read_csv(save_path)
    
    # upload merged dataframe to S3
    upload_dataframe_to_s3_as_parquet(df=df_fetched, bucket_name=bucket_name_nlp, file_key='sim_word2vec_records.parquet')
    
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