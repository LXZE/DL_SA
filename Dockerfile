FROM python:3.6-slim-stretch

# get apt gcc
RUN apt-get update \
	&& apt-get install -y --no-install-recommends gcc build-essential \
	&& rm -rf /var/lib/apt/lists/*

# install pip
RUN pip3 --no-cache-dir install \
	numpy \
	pandas \
	scikit-learn \
	gensim \
	keras \
	matplotlib \
	pythainlp

RUN	pip3 --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl

# delete gcc and cache
RUN apt-get purge -y --auto-remove gcc build-essential \
	&& rm -r /root/.cache

# set env
WORKDIR /