# Larger LSTM Network to Generate Haikus
import sys
import argparse
import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils


class HaikuGeneratorLSTM:
	def __init__(self, td_filename, nw_filename):
		self.td = td_filename         # Training corpus filename
		self.nw_path = nw_filename	  # Desired network weights path
		self.model = None
		self.X = None
		self.y = None
		self.word_to_index = None
		self.index_to_word = None

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
		X = np.zeros((len(word_phrase_pairs), n_unique_words, 1))
		y = np.zeros((len(word_phrase_pairs), n_unique_words))
		for i in range(0, len(word_phrase_pairs), 1):
			seq_in = word_phrase_pairs[i][0]
			seq_out = word_phrase_pairs[i][1]
			X[i][word_to_index[seq_in]] = 1
			for word in seq_out.split():				
				y[i][word_to_index[word]] = 1
				
		self.X = X
		self.y = y
		#print(X[0])
		#print(y[0])

		print("[TRAINING] Defining the LSTM based neural network at a word level:")
		# define the LSTM model (2-stacked lstm)	
		model = Sequential()
		model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
		model.add(Dropout(0.2))
		model.add(LSTM(256))
		model.add(Dropout(0.2))
		model.add(Dense(n_unique_words, activation='softmax'))

		model.compile(loss='categorical_crossentropy', optimizer='adam')

		if(train):

			# define the checkpoint
			print("[TRAINING] Fitting the model and recording checkpoints: ")
			filepath= self.nw_path + "-{epoch:02d}-{loss:.4f}.hdf5"
			checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
			callbacks_list = [checkpoint]

			# fit the model
			model.fit(X, y, epochs=20, batch_size=256, callbacks=callbacks_list)
			model.save_weights(self.nw_path, overwrite=True)
			self.model = model

			print("[TRAINING] Done.")

		else:
			
			try:
				model.load_weights(self.nw_path)
				model.compile(loss='categorical_crossentropy', optimizer='adam')

			except Exception as e:
				print("[ERROR] ", str(e))
			self.model = model
			print("[TRAINING] Model weights loaded.")

	def sample_word_lvl(self, queue):
		model = self.model
		word_to_index = self.word_to_index
		index_to_word = self.index_to_word	
		X = self.X
		y = self.y	
		
		for i in range(0, 3):
			queue_x = np.zeros((1, X.shape[1], 1))
			queue_x[0, word_to_index[queue]] = 1
			sampled_y = model.predict(queue_x, verbose=0)
			max_index = np.argmax(sampled_y)
			indices = np.argpartition(sampled_y[0], -4)[-4:]
			print(indices)
			phrase = ""
			for i, index in enumerate(indices):
				phrase += str(index_to_word[index])
				phrase += " "
			print(phrase)
			queue = index_to_word[max_index]
			print(queue)				

	def train_char_lvl(self, train=True):
		# load ascii text and convert to lowercase
		raw_text = open(self.td).read()
		raw_text = raw_text.lower()
		
		# create mapping of unique chars to integers
		chars = sorted(list(set(raw_text)))
		char_to_int = dict((c, i) for i, c in enumerate(chars))
		int_to_char = dict((i, c) for i, c in enumerate(chars))
		self.char_to_int = char_to_int
		self.int_to_char = int_to_char

		words = set(raw_text.split())

		# summarize the loaded data
		n_chars = len(raw_text)
		n_vocab = len(chars)
		n_words = len(words)

		print("[SETUP] Total # of Characters: ", n_chars)
		print("[SETUP] Total # of Unique Characters: ", n_vocab)
		print("[SETUP] Total # of Unique Words: ", n_words)
	
		# prepare the dataset of input to output pairs encoded as integers
		seq_length = 100
		dataX = []
		dataY = []
		for i in range(0, n_chars - seq_length, 1):
			seq_in = raw_text[i:i + seq_length]
			seq_out = raw_text[i + seq_length]
			dataX.append([char_to_int[char] for char in seq_in])
			dataY.append(char_to_int[seq_out])
		n_patterns = len(dataX)
		print("[SETUP] Total # of Patterns: ", n_patterns)

		# reshape X to be [samples, time steps, features]
		X = np.reshape(dataX, (n_patterns, seq_length, 1))
		# normalize
		X = X / float(n_vocab)
		# one hot encode the output variable
		y = np_utils.to_categorical(dataY)
		self.X = X
		self.y = y

		print("[TRAINING] Defining the LSTM based neural network:")
		# define the LSTM model (2-stacked lstm)	
		model = Sequential()
		model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
		model.add(Dropout(0.2))
		model.add(LSTM(256))
		model.add(Dropout(0.2))
		model.add(Dense(y.shape[1], activation='softmax'))
		model.compile(loss='categorical_crossentropy', optimizer='adam')
		
		if(train):

			# define the checkpoint
			print("[TRAINING] Fitting the model and recording checkpoints: ")
			filepath=self.nw_path + "-{epoch:02d}-{loss:.4f}.hdf5"
			checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
			callbacks_list = [checkpoint]

			# fit the model
			model.fit(X, y, epochs=10, batch_size=128, callbacks=callbacks_list)
			self.model = model

			print("[TRAINING] Done.")

		else:
			
			try:
				model.load_weights(self.nw_path)
				model.compile(loss='categorical_crossentropy', optimizer='adam')

			except Exception as e:
				print("[ERROR] ", str(e))
			self.model = model
			print("[TRAINING] Model weights loaded.")

	def sample_char_lvl(self):

		# load ascii text and convert to lowercase
		raw_text = open(self.td).read()
		raw_text = raw_text.lower()

		# create mapping of unique chars to integers, and a reverse mapping
		chars = sorted(list(set(raw_text)))
		char_to_int = dict((c, i) for i, c in enumerate(chars))
		int_to_char = dict((i, c) for i, c in enumerate(chars))

		# summarize the loaded data
		n_chars = len(raw_text)
		n_vocab = len(chars)
		print("[SETUP] Total Characters: ", n_chars)
		print("[SETUP] Total Vocab: ", n_vocab)

		# prepare the dataset of input to output pairs encoded as integers
		seq_length = 100
		dataX = []
		dataY = []
		for i in range(0, n_chars - seq_length, 1):
			seq_in = raw_text[i:i + seq_length]
			seq_out = raw_text[i + seq_length]
			dataX.append([char_to_int[char] for char in seq_in])
			dataY.append(char_to_int[seq_out])
		n_patterns = len(dataX)
		print("[SETUP] Total Patterns: ", n_patterns)

		# reshape X to be [samples, time steps, features]
		X = np.reshape(dataX, (n_patterns, seq_length, 1))
		# normalize
		X = X / float(n_vocab)
		# one hot encode the output variable
		y = np_utils.to_categorical(dataY)

		model = self.model

		# pick a random seed
		start = np.random.randint(0, len(X)-1)
		pattern = dataX[start]
		print("[DEBUG] Start: " + str(start))

		print("[GENERATION] SEED START:")
		print("\"", ''.join([int_to_char[value] for value in pattern]), "\"")
		print("[GENERATION] SEED END")

		# generate characters
		for i in range(1000):
			x = np.reshape(pattern, (1, len(pattern), 1))
			x = x / float(n_vocab)
			prediction = model.predict(x, verbose=0)
			index = np.argmax(prediction)
			result = int_to_char[index]
			seq_in = [int_to_char[value] for value in pattern]
			#print(result)
			sys.stdout.write(result)
			pattern.append(index)
			pattern = pattern[1:len(pattern)]

		print("\n [GENERATION] Done.")


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

	args = vars(ap.parse_args())
	nw_filename = args["network_weights"]
	data_filename = args["data"] # only for testing purposes
	mode = args["mode"]

	#print("[SETUP] Method: " + str(method))
	print("[SETUP] Training Data: " + str(data_filename)) 
	print("[SETUP] Network Weights Filename Path: " + str(nw_filename)) 

	lstm_NN = HaikuGeneratorLSTM(data_filename, nw_filename)

	if mode == 'train':
		lstm_NN.train_word_lvl(train=True)
	elif mode == 'sample':
		lstm_NN.train_word_lvl(train=False)
		lstm_NN.sample_word_lvl("clean")
