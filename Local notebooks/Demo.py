import boto3
import logging 
from botocore.exceptions import ClientError
import pandas as pd 
import io
import numpy as np 
# Import pre-trained word2vec models  
import gensim
import spacy
# In terminal, run python -m spacy download en_core_web_lg
nlp = spacy.load('en_core_web_lg')

##- Demo input values 
ref_row = 888 # metadata row id 
text_subject = "text_en_cleaned"  # text_en_cleaned or "title_en_cleaned"

#-- Read in the data 
def open_s3_file(bucket, filename):
    """Open a S3 parquet file from bucket and filename and return the parquet as pandas dataframe
    :param bucket: Bucket name
    :param filename: Specific file name to open
    :return: body of the file as a string
    """
    try: 
        buffer = io.BytesIO()
        s3 = boto3.resource('s3')
        object = s3.Object(bucket, filename)
        object.download_fileobj(buffer)
        df = pd.read_parquet(buffer)
        
        return df
    except ClientError as e:
        logging.error(e)
        return False

filename = "records_cleaned_for_ml.parquet"
bucket_name = "webpresence-search-similarity-data-dev"  
df = open_s3_file(bucket_name, filename)
print('Data shape is', df.shape)

#-- Prepare and slice the data for model comparison  a   
df = df[['features_properties_id', 'features_properties_title_en','title_en_cleaned', 'text_en_cleaned']]
df['id'] = range(0, df.shape[0])
#df_test = df.iloc[[193, 365,495, 1001, 1002, 1003, 1004, 1005, 1006, 1007]]
df_test = df


#-- Vectorization
ref_sent = df_test.loc[df_test['id']==ref_row, text_subject].iloc[0]
ref_sent_vec = nlp(ref_sent)
all_docs = [nlp(row) for row in df_test[text_subject]]  


#-- Calculate similarity score from Gensim 
sims = []
doc_id = []
for i in range(len(all_docs)):
    sim = all_docs[i].similarity(ref_sent_vec)
    sims.append(sim)
    doc_id.append(i)
    sims_docs = pd.DataFrame(list(zip(doc_id, sims)), columns = ['doc_id', 'sims'])
 
sims_docs_sorted = sims_docs.sort_values(by='sims', ascending=False)
#print("The type of sims_doc_sorted is {} \n The values of sims_doc_sorted is {}".format(type(sims_docs_sorted), sims_docs_sorted))

"""
# Pair-wise similarity calculation 
sims = []
for i in range(len(all_docs)):
    row = []
    for j in range(len(all_docs)):
        row.append(all_docs[i].similarity(all_docs[j]))
    sims.append(row)
sim_df = pd.DataFrame (sims, columns = ['col1', 'col2','col3','col4','col5','col6','col7','col8','col9','col10',])
print(sim_df)
sim_df.to_csv('sim.csv')
#data = data.drop(labels=0, axis=0)
"""

#-- Select top 5 similart sentence relate to the reference sentence  
top5_sim_docs = df_test.iloc[sims_docs_sorted['doc_id'][1:6]]
#print(df_test[df_test['id']==1001]['features_properties_title_en'].values)
#print('Top5_sim_docs are \n {}'.format(top5_sim_docs))

# Concatenate the top 5 text with it's similarity scores 
top_sim_scores= pd.concat([top5_sim_docs, sims_docs_sorted['sims'][1:6]], axis=1)
#print('The head of top_sim_scores is \n')
#print(top_sim_scores.head()) 

print('The reference metadata row id is {}  with an uuid {} \n'.format(df_test.loc[df_test['id']==ref_row, 'features_properties_title_en'], df_test.loc[df_test['id']==ref_row,'features_properties_id']))
print("Model traning is based on {}\n".format(text_subject))
for (text, sim, uuid) in zip (top_sim_scores['features_properties_title_en'], top_sim_scores['sims'], top_sim_scores['features_properties_id']):
    print('The top 5 similar metadata are :{}, with a cosine similarity scores of {:.2f}, and uuid is {}'.format(text, sim, uuid))
