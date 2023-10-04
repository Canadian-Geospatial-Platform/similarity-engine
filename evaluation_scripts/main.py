import pandas as pd
import numpy as np


data = pd.read_parquet("/home/rsaha/projects/similarity-engine/data/records_sep_18.parquet")

theme_to_keywords_mapping = {
    "Administration": ["boundaries", "planning_cadastre", "location"],
    "Economy": ["economy", "farming"],
    # "Emergency": [],
    "Environment": ["biota", "environment", "elevation", "inland_waters", "oceans", "climatologyMeteorologyAtmosphere"],
    "Imagery": ["imageryBaseMapsEarthCover"],
    "Infrastructure": ["structure", "transport", "utilitiesCommunication"],
    # "Legal": [],
    "Science": ["geoscientificInformation"],
    "Society": ["health", "society", "intelligenceMilitary"],
    # "foundation": [],
}

# Samples 2 records from the dataframe for each theme.
sampled_records = {}
for theme, value in theme_to_keywords_mapping.items():
    print("Theme: ", theme)
    records = data[data['features_properties_topicCategory'].str.contains('|'.join(value))].sample(2)
    sampled_records[theme] = records['features_properties_id'].values.tolist()
print("Sampled records: ", sampled_records)



# Read in the dynamodb_results.csv file for the 'Emergency' and 'Legal' categories.
dynamodb_results = pd.read_csv("/home/rsaha/projects/similarity-engine/data/dynamodb_results.csv")
for key in ['Legal', 'Emergency']:
    print("Key: ", key)
    records = dynamodb_results[dynamodb_results['tag'] == key.lower()].sample(2)
    sampled_records[key] = records['uuid'].values.tolist()

print('sampled_records: ', sampled_records)

all_uuids = []
for key, value in sampled_records.items():
    all_uuids.extend(value)

print(len(set(all_uuids)))