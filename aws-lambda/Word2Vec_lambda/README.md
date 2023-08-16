# Cloud Formation Package for Similarity Engine using Word2Vec

Follow the steps below to create the zip package:

1. Navigate to the directory where the `requirements.txt` is:
    ```
    cd similarity-engine-word2vec-model-dev/
    ```

2. Install the required packages:
    ```
    pip install -t similarity-engine-word2vec-model-build/ -r requirements.txt
    ```

3. Change to the build directory:
    ```
    cd similarity-engine-word2vec-model-build/
    ```

4. Zip the necessary files:
    ```
    zip -r similarity-engine-word2vec-model.zip ../app.py ../dynamodb.py ../__init.py__ ./*

    ```