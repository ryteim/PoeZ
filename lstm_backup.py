# Larger LSTM Network to Generate Haikus
import sys
import argparse
import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, RepeatVector
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import expand


class HaikuGeneratorLSTM:
	def __init__(self, td_filename, nw_filename):
		self.td = td_filename         # Training corpus filename
		self.nw_path = nw_filename	  # Desired network weights path
		self.model = None
		self.X = None
		self.y = None
		self.word_to_index = None
		self.index_to_word = None
		self.n_unique_words = None

	def TextDataGenerator(self, word_phrase_pairs, len_longest_phrase, n_unique_words, word_to_index):
		# Setup up input and output pairs:
		# Want input: word
		# Want output: poetic phrase
		# prepare the dataset of input to output pairs encoded as integers
		# one hot encode the inputs and outputs
		
		while 1:

			# word_phrase_pairs

			for wp in word_phrase_pairs:
				X = np.zeros((len(wp), 1, n_unique_words))
				y = np.zeros((len(wp), len_longest_phrase, n_unique_words))
				
				seq_in = wp[0]
				seq_out = wp[1]
				X[0][0][word_to_index[seq_in]] = 1
				for idx, word in enumerate(seq_out.split()):				
					y[0][idx][word_to_index[word]] = 1
						
				self.X = X
				self.y = y

				yield (X, y)


	def train_word_lvl(self, train=True):
		# load ascii text and convert to lowercase
		raw_text = open(self.td, 'r').readlines()
		raw_text = [row.lower() for row in raw_text]
		
		# df = pd.read_csv(self.td, names=['word1', 'phrase1', 'word2', 'phrase2', 'word3', 'phrase3'], headers=None)
				
		# create mapping of unique chars to integers
		all_words = []
		word_phrase_pairs = []
		for row in raw_text:			
			row_words = row.split(',')
			row_words = [word.strip() for word in row_words]
			word1 = row_words[0]
			word2 = row_words[2]			
			word3 = row_words[4]
			phrase1_words = row_words[1].split()
			phrase2_words = row_words[3].split()
			phrase3_words = row_words[5].split()
			all_words.extend([word1, word2, word3])
			all_words.extend(phrase1_words)
			all_words.extend(phrase2_words)
			all_words.extend(phrase3_words)
			word_phrase_pairs.append((row_words[0], row_words[1]))
			word_phrase_pairs.append((row_words[2], row_words[3]))
			word_phrase_pairs.append((row_words[4], row_words[5]))
		

		words = set(all_words)
		word_phrase_pairs = list(set(word_phrase_pairs))
		word_to_index = dict((w,i) for i, w in enumerate(words))
		index_to_word = dict((i,w) for i, w in enumerate(words))
		self.word_to_index = word_to_index
		self.index_to_word = index_to_word

		# summarize the loaded data
		n_unique_words = len(words)
		self.n_unique_words = n_unique_words
		n_words = len(all_words)
		len_longest_phrase = 0
		print("[SETUP] Total # of Words: " + str(n_words))
		print("[SETUP] Total # of Unique Words: " + str(n_unique_words))


		for wp_pair in word_phrase_pairs:
			if len(wp_pair[1].split()) > len_longest_phrase:
				len_longest_phrase = len(wp_pair[1].split())
				longest_phrase = wp_pair[1]

		print("[SETUP] Longest Sentence in # of Words: " + str(len_longest_phrase))
		print("[SETUP] Longest Sentence: " + str(longest_phrase))
		

		# REMOVED SHIT HERE
		n_words = len(all_words)

		print("[SETUP] Total # of Words: ", n_words)
		print("[SETUP] Total # of Unique Words: ", n_unique_words)
		#print(word_to_index)
		#print(index_to_word)

		# Setup up input and output pairs:
		# Want input: word
		# Want output: poetic phrase
		# prepare the dataset of input to output pairs encoded as integers
		# one hot encode the inputs and outputs

		# X = np.zeros((len(word_phrase_pairs), 1, n_unique_words))
		# y = np.zeros((len(word_phrase_pairs), len_longest_phrase, n_unique_words))
		
		# print("[TRAINING] Shapes: ")
		# print("[TRAINING] Shape of X: " + str(X.shape))
		# print("[TRAINING] Shape of y: " + str(y.shape))
		# for i in range(0, len(word_phrase_pairs), 1):
		# 	seq_in = word_phrase_pairs[i][0]
		# 	seq_out = word_phrase_pairs[i][1]
		# 	X[i][0][word_to_index[seq_in]] = 1
		# 	for idx, word in enumerate(seq_out.split()):				
		# 		y[i][idx][word_to_index[word]] = 1
				
		# self.X = X
		# self.y = y
		#print(X[0])
		#print(y[0])    
		# END OF REMOVED SHIT HERE
		print("[TRAINING] Defining the LSTM based neural network at a word level:")
		# define the LSTM model (2-stacked lstm)	
		model = Sequential()
		model.add(LSTM(256, input_shape=(1, n_unique_words), return_sequences=True))
		model.add(Dropout(0.2))
		model.add(LSTM(256))
		model.add(RepeatVector(len_longest_phrase))		
		model.add(Dropout(0.2))		
		model.add(LSTM(256, return_sequences=True))
		model.add(Dropout(0.2))
		# model.add(Dense(n_unique_words, activation='softmax'))
		# model.add(RepeatVector(len_longest_phrase))
		model.add(TimeDistributed(Dense(n_unique_words, activation='softmax'), input_shape=(256, len_longest_phrase)))

		print("[TRAINING][DEBUG] Model summary: ")
		model.summary()
		print("[TRAINING][DEBUG] Inputs: {}".format(model.input_shape))
		print("[TRAINING][DEBUG] Outputs: {}".format(model.output_shape))

		model.compile(loss='categorical_crossentropy', optimizer='adam')

		if(train):

			# define the checkpoint
			print("[TRAINING] Fitting the model and recording checkpoints: ")
			filepath= self.nw_path + "-{epoch:02d}-{loss:.4f}.hdf5"
			checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
			callbacks_list = [checkpoint]

			# fit the model
			model.fit_generator(self.TextDataGenerator(word_phrase_pairs, len_longest_phrase, n_unique_words, word_to_index), steps_per_epoch=1000, epochs=10, verbose=0,callbacks=callbacks_list)
			# model.fit(X, y, epochs=50, batch_size=32, callbacks=callbacks_list)
			model.save_weights(self.nw_path, overwrite=True)
			self.model = model

			print("[TRAINING] Done.")

		else:
			
			try:
				model.load_weights(self.nw_path)
				model.compile(loss='categorical_crossentropy', optimizer='adam')

			except Exception as e:
				print("[ERROR] " + str(e))
			self.model = model
			print("[TRAINING] Model weights loaded.")

	def sample_word_lvl(self, queue):
		model = self.model
		word_to_index = self.word_to_index
		index_to_word = self.index_to_word	
		X = self.X
		y = self.y	
		
		queue_x = np.zeros((1, 1, self.n_unique_words))
		try: 				
			queue_x[0][0][word_to_index[queue]] = 1
		except Exception as e:
			print("[ERROR] " + str(e))
			print("[ERROR] Likely that the query is not recognized.")
			return
		
		sampled_y = model.predict(queue_x, verbose=0)
		phrase = ""
		max_index = 0
		for i in range(0, sampled_y.shape[1]):
			max_index = np.argmax(sampled_y[0][i])
			indices = np.argpartition(sampled_y[0][i], -4)[-4:]
			# print(indices)
			phrase2 = ""
			for j, index in enumerate(indices):
				phrase2 += str(index_to_word[index])
				phrase2 += " "
			print(phrase2)
			phrase += str(index_to_word[max_index])
			phrase += " "
		
		# print("[OUTPUT] Phrase 1: " + str(phrase))
		# queue = index_to_word[max_index]
		# print("[OUTPUT] New Queue: " + str(queue))
		return phrase				

	def __str__(self):
		'''
		For printing purposes.
		'''
		return '%s%s' % ("LSTM trained on: ", self.td)


if __name__ == '__main__':

	# terminal argument parser
	ap = argparse.ArgumentParser()
	#list_of_methods = ['wordnet', 'word2vec', 'onehot', 'glove']
	list_of_modes = ['train', 'sample']
	#ap.add_argument("-m", "--method", required=False, help="Method to use for WSD. Default = wordnet.", default="wordnet", choices = list_of_methods)
	ap.add_argument("-d", "--data", required=False, help="Training data corpus to train on.", default="all_words-wordnet.txt")
	ap.add_argument("-nw", "--network-weights", required=True, help="Filename selected for saving the network weights.")
	ap.add_argument("-m", "--mode", required=True, help="Choose between training mode or sampling mode.", default="train", choices=list_of_modes)
	ap.add_argument("-q", "--query", required=False, help="Query word to generaet a poem about.", default="sing")

	args = vars(ap.parse_args())
	nw_filename = args["network_weights"]
	data_filename = args["data"] # only for testing purposes
	mode = args["mode"]
	query = args["query"]

	#print("[SETUP] Method: " + str(method))
	print("[SETUP] Training Data: " + str(data_filename)) 
	print("[SETUP] Network Weights Filename Path: " + str(nw_filename)) 

	lstm_NN = HaikuGeneratorLSTM(data_filename, nw_filename)

	if mode == 'train':
		print("[SETUP] Training mode.")
		lstm_NN.train_word_lvl(train=True)
	elif mode == 'sample':
		print("[SETUP] Sampling mode.")
		print("[SETUP] Query word: " + str(query))
		lstm_NN.train_word_lvl(train=False)

		topics = expand.expand(str(query), 'glove_haiku_50')	
		print("[OUTPUT] Topics: ")
		print(topics)
		q2 = topics[1]
		q3 = topics[2]

		p1 = lstm_NN.sample_word_lvl(query)
		p2 = lstm_NN.sample_word_lvl(q2)
		p3 = lstm_NN.sample_word_lvl(q3)
		print("[OUTPUT] Final poem: ")
		print(p1)
		print(p2)
		print(p3)

		