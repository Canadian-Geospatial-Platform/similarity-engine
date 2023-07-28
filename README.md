# similarity_engine
 Machine learning models to recommend similary uuid on geo.ca


## Setup 
Having a virtual environment using anaconda or virtualenv will help project specific scope of packages.

The project uses python 3.9.17 inside an anaconda environment


```
conda create -n similarity_engine python=3.9.17
```

Install packages.
```
pip install -r requirements.txt
```

## Models

The following models are being used.

1. Word2Vec
2. BERT
3. RoBERTa - base models
4. stsb-roberta-large


## Evaluation
Using perplexity for now. May switch to some external metrics.