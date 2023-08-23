# Navigate to the directory containing `requirements.txt`
cd similarity-engine-data-process/

# Install required packages
pip install -t similarity-engine-data-preprocess-20230822-2200/ -r requirements.txt

# Change to the build directory
cd similarity-engine-data-preprocess-20230822-2200/

# Zip the necessary files
zip -r similarity-engine-data-preprocess-20230822-2200.zip ../app.py ../__init.py__ ./*
