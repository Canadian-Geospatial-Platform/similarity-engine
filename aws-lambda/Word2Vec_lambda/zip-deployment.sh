#!/bin/bash
# Build Cloud Formation Package for Similarity Engine using Word2Vec
# Author: Xinli Cai 
# Date: August 29, 2023 


# Cloud Formation Package for Similarity Engine using Word2Vec

# Step 1: Navigate to the directory where the `requirements.txt` is
cd similarity-engine-word2vec-model-dev/

# Step 2: Install the required packages
pip install -t similarity-engine-word2vec-model-build/ -r requirements.txt

# Step 3: Change to the build directory
cd similarity-engine-word2vec-model-build/

# Step 4: Zip the necessary files
zip -r similarity-engine-word2vec-model.zip ../app.py ../dynamodb.py ../__init.py__ ./*

# Restore to the initial directory
cd ..
