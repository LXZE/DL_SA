DL_SA (Deep learning on Sentiment Analysis)
=====

The sentiment classifier model based on State-of-the-art deep learning model, implemented with keras.

This repository also include the proprocessing code which working with Twitter data and Thai sentence.

---
## Requirements
- python 3.6 ++

- pandas
- numpy
- gensim
- pythainlp
- tensorflow
- keras
- sklearn
- tweepy

---
## Files list
- `clean.py`  
   Cleaning the text data, use as a library in this project.

- `data2npvec.py`  
   Convert cleaned data to token, then save list of tokens in pkl format. (Having 2 modes, default and no-tensor, default one uses tf to import and cut the sentence, no-tensor uses newmm algorithm from pythainlp)

- `filter.py`  
   filter the cleaned data and save according to entity data in `entity.txt`, file will saved in `data/filtered/` with first column in `entity.txt`

- `gather_data.py`  
   gather the results from `filter.py` which stored in `data/filtered` directory then accumulate all of data and save into one file for each entity in `entity.txt`

- `manual_classify.py`  
   This code import pythainlp sentiment classifier which used for pre-classifying unlabeled twitter data, then I manually classify the result later for ease of work.

- `pipeline.py`  
   Combining `twit_reader.py`, `filter.py` and `gather_data.py` to work along in one file. This works with every file in `data/` directory

- `polar2xy.py`  
   Testing and validating the data's shape and converting method before moving the code to google's colab.

- `read_wongnaidata.py`  
   Converting data from wongnai's directory and split the bad and good review to negative and positive data respectively.

- `test_model.py`  
   Testing the result model from google's colab, randomly picking some of sentence and then try to predict the sentiment polarity.

- `test_vector.py`  
   Converting token into list of integer, according to word list of pre-trained vector matrix.

- `twit_crawl.py`  
   Crawling tweet from twitter, credentials should stored in `account.ini` which contains access_token and consumer_key with secret. Result with stored in `txt/` directory.

- `twit_reader.py`  
   Read the data from given argument file (used by `pipeline.py`) then process and clean in parallel, then save into `data/clean/` directory.

---

## Involved repository & project
- [word_fixer][1]

   A lookup table I've created for fixing word error after segmentation.

- [thai-word-segmentation][2]
- [sentiment_analysis_thai][3]
- [lexicon-thai][4]

[1]: https://github.com/LXZE/Thai_word_fixer
[2]: https://github.com/sertiscorp/thai-word-segmentation/tree/5c77e020d592eef38c20a89a81e1c3eb957ecac8
[3]: https://github.com/JagerV3/sentiment_analysis_thai/tree/b99ef710e01691e2b8d8555c38f3ef418397ec4e
[4]: https://github.com/PyThaiNLP/lexicon-thai/tree/eb2f26e6cf5f5e94bb560a742f027c9f6a5354a8

