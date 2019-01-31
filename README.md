DL_SA (Deep learning on Sentiment Analysis)
=====

The thai online text sentiment classifier model based on deep learning model, implemented with keras.

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

- `data2token.py`  
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

- `twit_crawl.py`  
   Crawling tweet from twitter, credentials should stored in `account.ini` which contains access_token and consumer_key with secret. Result with stored in `txt/` directory.

- `twit_reader.py`  
   Read the data from given argument file (used by `pipeline.py`) then process and clean in parallel, then save into `data/clean/` directory.

---

## Involved repository & project
- [word_fixer][1]

   A lookup table I've created for fixing word error after segmentation.

- [thai-word-segmentation][2]

[1]: https://github.com/LXZE/Thai_word_fixer
[2]: https://github.com/sertiscorp/thai-word-segmentation/tree/5c77e020d592eef38c20a89a81e1c3eb957ecac8