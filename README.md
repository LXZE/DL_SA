DL_SA (Deep learning on Sentiment Analysis)
=====

The repository of thesis "Thai Comment Sentiment Analysis on Social Networks with Deep Learning Approach" 

Supervised by: [Asst. Prof. Jumpol Polvichai, Ph.D.](http://cpe.kmutt.ac.th/en/staff-detail/Jumpol) and [Asst. Prof. Nuttanart Facundes, Ph.D.](http://cpe.kmutt.ac.th/en/staff-detail/Nuttanart)

This repository mainly contains code for performing sentiment prediction and notebook files for recording the training process and some visualization.

Also, it includes tweet crawler and text proprocessing code which work with Twitter data and Thai sentence specifically.

## Thesis abstract

> In recent years, many people have posted their comments publicly on social media or websites. As a result, there is a large amount of text data available that could be analysed to gain some insights from users, which can be done by Sentiment Analysis. However, analysing human language data regarding its semantics is difficult because the machine does not have prior knowledge in a language. Therefore, Deep Learning techniques are introduced, as it shows the effectiveness in analysing a massive amount of data. Several applied Deep Learning works have experimented on English and show a satisfying result. Thus, it is an opportunity to explore and perform Sentiment Analysis on Thai online documents using Natural Language Processing techniques and Deep Learning algorithms. There are two main problems to be solved in this work. The first problem is the word ambiguity manipulation, and the other task is an automatic sentiment classification. This work shows our process of Thai online document cleansing to handle errors in raw Thai texts. Also, this work describes the entire process, from data collection, experimentation, evaluation, to the findings of the most suitable Deep Learning algorithm to classify the sentiment polarity from a given document. The results show that every Deep Learning model yields high accuracy and has relatively similar performance of sentiment classification, and a model using the One-dimensional Convolutional Neural Network requires the least time to train. The results can be used for future development in Thai Natural Language Processing and Sentiment Analysis.

> Keywords: 	Deep Learning/ Natural Language Processing/ Sentiment Analysis/ Thai Language

## Slide
- [ Paper presentation at ITC-CSCC 2019, Jeju, Korea ]( https://docs.google.com/presentation/d/1Eycqslpurvfdls4i8BB4gkl0uhpGFtWh3e4H-MXnhQ0/edit?usp=sharing )
- [ The final presentation for Master's degree thesis ]( https://docs.google.com/presentation/d/1OWGwTtOXKOxSEfaWXUH3842NLhqRBxwJdPbRw9J9Iww/edit?usp=sharing )

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

## Involved repository & project
- [thai-word-segmentation](https://github.com/sertiscorp/thai-word-segmentation/tree/5c77e020d592eef38c20a89a81e1c3eb957ecac8)

<!---
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
-->
