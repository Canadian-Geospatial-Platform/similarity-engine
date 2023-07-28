Create zip package for cloud formation 

cd similarity-engine-word2vec-model-dev/ (changd path to where the requirement.txt is)
pip install -t similarity-engine-word2vec-model-build/ -r requirements.txt
cd similarity-engine-word2vec-model-build/
zip -r similarity-engine-word2vec-model.zip ../app.py ../__init.py__ ./*