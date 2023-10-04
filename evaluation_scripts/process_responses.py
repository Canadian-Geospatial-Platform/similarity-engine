import ast
import numpy as np
import pandas as pd
import sys


import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


sys.path.append("/home/rsaha/projects/similarity-engine/src/")
from utils.data_loading import load_processed_parquet

from constants import category_mappings, random_category_mappings, model_order, \
    model_order_list, choice_mapping_order
    

responses = pd.read_excel("/home/rsaha/projects/similarity-engine/form_builder_data/Form builder template for model voting.xlsx")


# Do some processing.
responses = responses.drop([
    'Timestamp',
    'Email Address',
], axis=1)


modified_responses = responses.copy(deep=True)


# Replace the responses for all the 20 questions with the original choice mapping


for q in range(20):
    for row in range(responses.shape[0]):
        # For each row, change value to the actual choice according to the choice mapping order.
        modified_responses.iloc[row, q] = choice_mapping_order[q][int(responses.iloc[row, q]) - 1]  # Subtracting -1 becaus the options start from 1 but indices start from 0.


modified_responses.to_excel("/home/rsaha/projects/similarity-engine/form_builder_data/modified_responses_original_mapping.xlsx")