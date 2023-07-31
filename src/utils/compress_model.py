import gensim
import compress_fasttext

import fasttext.util

fasttext.util.download_model('en', if_exists='ignore')  # English
big_model = fasttext.load_model('cc.en.300.bin')
# big_model = gensim.models.fasttext.FastTextKeyedVectors.load('/Users/simpleparadox/Documents/comlam_raw/word2vec.bin.gz')
small_model = compress_fasttext.prune_ft_freq(big_model, pq=True)
small_model.save('../models/small_w2v.bin')