# A global file for transformers fine-tuning.
# The model that will be fine-tuned depends on the command line arguments.


import argparse
import os
import sys
import time
import torch
import numpy as np

import wandb
import platform
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

# from custom_dataset import CustomDataset
from custom_model import CustomModel

sys.path.append("src/")

from utils.data_loading import load_processed_parquet, upload_dataframe_to_s3_as_parquet
from datasets import Dataset


argparse = argparse.ArgumentParser()

# Add arguments to the parser.
argparse.add_argument('--model_name', type=str, default='bert-base-uncased')
argparse.add_argument('--model_type', type=str, default='bert')
argparse.add_argument('--batch_size', type=int, default=8)
argparse.add_argument('--epochs', type=int, default=20)
argparse.add_argument('--lr', type=float, default=2e-5)
argparse.add_argument('--load_model_path', type=str, default='')
argparse.add_argument('--train_on_full_data', type=bool, default=True)
argparse.add_argument('--load_from_aws', type=bool, default=False)


args = argparse.parse_args()

# Set the device.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# wandb login.
print("Device: ", device)

args.device = device
if args.train_on_full_data == 'True':
    args.train_on_full_data = True
print("Train on full data: ", args.train_on_full_data)


# Initialize wandb and login.
# wandb.init(project='geo.ca')

args.load_model_path = "/home/rsaha/projects/similarity-engine/saved_models/trainer_bert_fine_tune_full_data/checkpoint-500/"
args.output_dir = "/home/rsaha/projects/similarity-engine/saved_models/trainer_bert_fine_tune_full_data"


# Load the data.
if args.load_from_aws:
    df = load_processed_parquet()
else:
    if platform.system() == 'Windows':
        df = pd.read_csv('D:\\similarity-engine\\notebooks\\df_training_full.csv')
    else:
        df = pd.read_csv('/home/rsaha/projects/similarity-engine/notebooks/df_training_full.csv')

df = df.fillna('')
df['merged'] = df['features_properties_title_en'] + ' ' + df['metadata_en_processed']

# Divide the data into train and test set. 
train_set = df.sample(frac=0.9, random_state=42)  # Fixing the seed to 42 to reproducibility.
test_set = df.drop(train_set.index)


if args.train_on_full_data:
    train_set = df
    test_set = df
train_dataset = Dataset.from_pandas(train_set)
test_dataset = Dataset.from_pandas(test_set)

# Initialize the model.
model = CustomModel(args, load_model_from_path=True, model_path=args.load_model_path)

# loss = model.hf_trainer_evaluate(test_dataset)

# Train the model.
# model.trainer_finetune(train_dataset, test_dataset, evaluate_only=False, evaluate_while_training=False)

# # Get the embeddings for the texts in the test set.
# df['embeddings'] = df['merged'].apply(model.get_embeddings)
# np.savez_compressed('/home/rsaha/projects/similarity-engine/data/all_fine_tuned_bert_embeds.npz', all_cls_embeddings)
# # model.get_embeddings('hello world')
# "01e01816-9f08-49eb-a5ac-7480195d90d4"
# # Calculate the similarity matrix.
# similarity_matrix = cosine_similarity(np.vstack(df['embeddings']))
# df['top_10_similar'] = [list(df.iloc[np.argsort(-row)][1:11].index) for row in similarity_matrix]
# df['top_20_similar'] = [list(df.iloc[np.argsort(-row)][1:21].index) for row in similarity_matrix]
# df = df.drop(columns=['embeddings'])
# df.to_csv("/home/rsaha/projects/similarity-engine/data/train_fine_tune_full_data_sim_matrix_no_embeds.csv")

df = pd.read_csv("/home/rsaha/projects/similarity-engine/data/train_fine_tune_full_data_sim_matrix_no_embeds.csv")

# Upload the data to S3.
upload_dataframe_to_s3_as_parquet(df, bucket_name='nlp-data-preprocessing', file_key='fine_tune_full_data_sim_matrix_no_embeds.parquet')
# np.savez_compressed('/home/rsaha/projects/similarity-engine/data/df_bert_cls_similarity_matrix.npz', similarity_matrix)

# Save model
# Save the embeddings and the similarity matrix to AWS S3 bucket.
