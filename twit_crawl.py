# -*- coding: utf-8 -*-

# TODO: Read extend tweet with link which isn't rt and quoted 

import sys, os
import codecs
import time as t
# sys.stdout = codecs.getwriter('utf8')(sys.stdout)
# sys.stderr = codecs.getwriter('utf8')(sys.stderr)

import configparser
import io

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

import signal
import sys

import json
import pprint

import codecs

config = configparser.ConfigParser()
config.read('account.ini')

access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']
consumer_key = config['twitter']['consumer_key']
consumer_secret = config['twitter']['consumer_secret']

ftime = lambda: t.strftime('%Y_%m_%d')
now = lambda: t.strftime('%Y/%m/%d-%H:%M:%S')
fileName = 'txt/twit_{}.txt'
errorFileName = 'error.txt'
file = codecs.open(fileName.format(ftime()), 'a', 'utf-8-sig')
errorFile = codecs.open(errorFileName, 'a', 'utf-8-sig')
baseTime = ftime()
def writeToFile(txt):
	global baseTime
	global file
	newTime = ftime()
	if baseTime != newTime:
		file.close()
		file = codecs.open(fileName.format(newTime), 'a', 'utf-8-sig')
		baseTime = newTime
	file.write(txt)
	file.write('\n[{}],\n'.format(now()))

def writeError(error, sleepTime = None):
	if sleepTime is None:
		sleepTime = 5
	global errorFile
	errorFile.write('[ERR] Time:{} Text:{}'.format(t.strftime('%Y/%m/%d-%H:%M:%S'), error))
	errorFile.write('\n')
	t.sleep(sleepTime)

def closeFile():
	print('Closing all opened files')
	global file
	global errorFile
	if file and errorFile:
		file.close()
		errorFile.close()

def signal_handler(signal, frame):
	print('Exit...')
	global file
	global errorFile
	global KB
	if file:
		file.close()
		errorFile.close()
		print('File Closed')
		KB = True
	sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

pp = pprint.PrettyPrinter()
def _print(txt):
	pp.pprint(txt)

def print_format(user, text):
	print('@%s: %s' % (user, text))

class StdOutListener(StreamListener):
	def on_data(self, data):
		txt = json.loads(data)
		try:
			tmp = txt
			res = '@%s: %s' % (txt['user']['screen_name'], txt['text'])
			txt = txt['text']
			if txt[-1] == u'…':	
				pass
			elif txt[:2] == 'RT':
				pass
			else:
				if txt.find(u'…') != -1:
					try:
						txt = tmp['extended_tweet']['full_text']
					except Exception as e:
						# Post from external service, can't retrieve full text
						print('Error:', e)
						# writeError(e, 0)
						# print(tmp.keys())
						# _print(tmp)
				print_format(tmp['user']['screen_name'], txt)
				writeToFile(txt)
				print('-'*30)
		except Exception as e:
			print('Error:', e)
			print_format('USER_TWIT_ERROR', txt)
			print('-'*30)
			# writeError(e, 0)
			# _print(tmp)
		return True

	def on_error(self, err):
		print('[ERR] in StdOutListener with error', err)
		writeError(err, 0)
		return False
if __name__ == '__main__':

	words = []
	with open('words-th.txt') as f:
		words = f.readlines()
	words = map(lambda word: word[:-1] ,words)

	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	l = StdOutListener()
	KB = False
	try:
		print('Start streaming connection')
		stream = Stream(auth, l)
		stream.filter(languages=['th'], track=words, stall_warnings= True)
	except Exception as e:
		print('[ERR] Error in while loop')
		writeError(e)
	finally:
		if not KB:
			print('Restart Program')
			writeError('Restart Program due to connection error', 0)
			closeFile()
			t.sleep(10)
			os.execv(sys.executable, ['/usr/bin/python3.6'] + sys.argv)