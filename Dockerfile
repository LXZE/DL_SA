FROM python:3.6-stretch

# install pip
RUN pip3 --no-cache-dir install \
	numpy \
	pandas \
	scikit-learn \
	gensim \
	keras \
	matplotlib \
	pythainlp

RUN	pip3 install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.12.0-cp36-cp36m-linux_x86_64.whl

RUN rm -r /root/.cache

# set env
WORKDIR /