# Larger LSTM Network to Generate Haikus
import sys
import argparse
import numpy as np
from scipy import spatial
import re
import string

from keras.models import Sequential
from keras.layers import Dense, TimeDistributed, RepeatVector
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

import expand
import word2vec

class Glove:
	def __init__(self, training_file):
		self.training_file = training_file
		self.vectors = []
		self.vocab = []
		self.model = dict()
		self.load_glove(training_file)

	def load_glove(self, training_file):
		raw = open(training_file, 'r').readlines()
		
		for line in raw:			
			split_line = line.split(' ')
			word = split_line[0]
			vector = split_line[1:]
			vector = [float(val) for val in vector]
			vector = np.array(vector)
			
			self.vocab.append(word)
			self.vectors.append(vector)
			self.model[word] = vector
			
		return


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


	def TextDataGenerator(self, word_phrase_pairs, len_longest_phrase, n_unique_words, word_to_index, embedding):
		# Setup up input and output pairs:
		# Want input: word
		# Want output: poetic phrase
		# prepare the dataset of input to output pairs encoded as integers
		# one hot encode the inputs and outputs
		
		# load word2vec model
		w2v_model = word2vec.load('./word2vec_training/text8.bin')
		glove_model = Glove('./glove_training/all_words.glove.100.txt')


		while 1:

			# word_phrase_pairs

			for wp in word_phrase_pairs:

				if embedding == 'onehot':

					X = np.zeros((len(wp), 1, n_unique_words))
					y = np.zeros((len(wp), len_longest_phrase+1, n_unique_words))
					# print(X[0])
					# print(y[0])

					seq_in = wp[0]
					seq_out = wp[1]
					seq_out_split = seq_out.split()
					len_diff = (len_longest_phrase+1) - len(seq_out_split)

					for i in range(0,len_diff):
						seq_out_split.append("stop")

					X[0][0][word_to_index[seq_in]] = 1
					for idx, word in enumerate(seq_out_split):				
						y[0][idx][word_to_index[word]] = 1

				elif embedding == 'word2vec':
					
					X = np.zeros((len(wp), 1, w2v_model.vectors.shape[1]))
					y = np.zeros((len(wp), len_longest_phrase+1, w2v_model.vectors.shape[1]))


					seq_in = wp[0]
					seq_out = wp[1]
					seq_out_split = seq_out.split()
					len_diff = (len_longest_phrase+1) - len(seq_out_split)

					for i in range(0,len_diff):
						seq_out_split.append("stop")
					
					try:
						X[0][0] = w2v_model[seq_in]
					except:
						# print("\nWord not modeled in input: " + str(seq_in))
						continue
					for idx, word in enumerate(seq_out_split):				
						try:
							y[0][idx] = w2v_model[word]	
						except: 
							# print("Word not modeled in target: " + str(word))
							break

				elif embedding == 'glove':
					X = np.zeros((len(wp), 1, glove_model.vectors[0].shape[0]))
					y = np.zeros((len(wp), len_longest_phrase+1, glove_model.vectors[0].shape[0]))


					seq_in = wp[0]
					seq_out = wp[1]
					seq_out_split = seq_out.split()
					len_diff = (len_longest_phrase+1) - len(seq_out_split)

					for i in range(0,len_diff):
						seq_out_split.append("<ENDLINE>")
					
					try:
						X[0][0] = glove_model.model[seq_in]
					except:
						# print("\nWord not modeled in input: " + str(seq_in))
						continue
					for idx, word in enumerate(seq_out_split):				
						try:
							y[0][idx] = glove_model.model[word]	
						except: 
							# print("Word not modeled in target: " + str(word))
							break

				# self.X.append(X)
				# self.y.append(y)
				yield (X, y)



	def clean(self,tokens):
		filtered_tokens = [word for word in tokens if word not in ["|", "'", ".", "!", "?", "-", "''", ",", "``", "(", ")", "[", "]", "--", "...", "....", "..", ";", ":"]]
		filtered_tokens = [word for word in filtered_tokens if word not in ["'ll", "'m", "'d", "'re", "'ve", "'s", "n't", "ca"]]
		filtered_tokens = [word.strip() for word in filtered_tokens]	# remove all whitespace from each word (before and after)
		filtered_tokens = [word.strip('.') for word in filtered_tokens]	# remove all whitespace from each word (before and after)
		filtered_tokens = [word.strip("'") for word in filtered_tokens]	# remove all whitespace from each word (before and after)
		filtered_tokens = [word.strip('"') for word in filtered_tokens]	# remove all whitespace from each word (before and after)
					
		return filtered_tokens




	def train_word_lvl(self, embedding, train=True):
		# load ascii text and convert to lowercase
		raw_text = open(self.td, 'r').readlines()
		raw_text = [row.lower() for row in raw_text]	
		# df = pd.read_csv(self.td, names=['word1', 'phrase1', 'word2', 'phrase2', 'word3', 'phrase3'], headers=None)
				
		# create mapping of unique chars to integers
		all_words = []
		word_phrase_pairs = []
		exclude = set(string.punctuation)
		for row in raw_text:			
			row_words = row.split(',')							# split by comma
			row_words = self.clean(row_words)
			# row_words = ''.join(ch for ch in row_words if ch not in exclude)
			# word1 = self.clean(row_words[0], 0)
			# word2 = self.clean(row_words[2], 0)			
			# word3 = self.clean(row_words[4], 0)
			# print(row_words)
			word1 = row_words[0]
			word2 = row_words[2]		
			word3 = row_words[4]
			phrase1_words = row_words[1].split()
			phrase2_words = row_words[3].split()
			phrase3_words = row_words[5].split()
			all_words.extend([word1, word2, word3, "stop", "<ENDLINE>"])
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
		# load word2vec model
		w2v_model = word2vec.load('./word2vec_training/text8.bin')
		glove_model = Glove('./glove_training/all_words.glove.100.txt')
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
		

		# n_words = len(all_words)

		# print("[SETUP] Total # of Words: ", n_words)
		# print("[SETUP] Total # of Unique Words: ", n_unique_words)
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
		

		print("[TRAINING] Defining the LSTM based neural network at a word level:")
		# define the LSTM model (2-stacked lstm)
		if embedding == 'onehot': 
			in_shape = (1, n_unique_words)
		elif embedding == 'word2vec':
			in_shape = (1, w2v_model.vectors.shape[1])
		elif embedding == 'glove':
			in_shape = (1, 100)		
		model = Sequential()

		# Working
		model.add(LSTM(256, input_shape=in_shape, return_sequences=True))
		model.add(Dropout(0.2))
		model.add(LSTM(256))
		model.add(RepeatVector(len_longest_phrase+1))		
		model.add(Dropout(0.2))		
		model.add(LSTM(256, return_sequences=True))
		model.add(Dropout(0.2))
		# model.add(Dense(n_unique_words, activation='softmax'))
		# model.add(RepeatVector(len_longest_phrase))
		if embedding == 'onehot':
			model.add(TimeDistributed(Dense(in_shape[1], activation='softmax'), input_shape=(256, len_longest_phrase+1)))
		elif embedding == 'word2vec':
			model.add(TimeDistributed(Dense(in_shape[1], activation='tanh'), input_shape=(256, len_longest_phrase+1)))
		elif embedding == 'glove':
			model.add(TimeDistributed(Dense(in_shape[1], activation='linear'), input_shape=(256, len_longest_phrase+1)))
	
		# Experimental (inputs (*, ))
		# model.add(LSTM(256, input_shape=in_shape, return_sequences=True))
		# model.add(Dropout(0.2))
		# model.add(LSTM(256))
		# model.add(Dropout(0.2))		
		# # model.add(Dense(n_unique_words, activation='softmax'))
		# # model.add(RepeatVector(len_longest_phrase))
		# if embedding == 'onehot':
		# 	model.add(Dense(in_shape[1], activation='softmax'))
		# elif embedding == 'word2vec':
		# 	model.add(Dense(in_shape[1], activation='softmax'))
		
		print("[TRAINING][DEBUG] Model summary: ")
		model.summary()
		print("[TRAINING][DEBUG] Inputs: {}".format(model.input_shape))
		print("[TRAINING][DEBUG] Outputs: {}".format(model.output_shape))

		if embedding == 'onehot':
			model.compile(loss='categorical_crossentropy', optimizer='adam')
		elif embedding == 'word2vec':
			model.compile(loss='mean_squared_error', optimizer='adam')
		elif embedding == 'glove':
			model.compile(loss='mean_squared_error', optimizer='adam')
		
		if(train):

			# define the checkpoint
			print("[TRAINING] Fitting the model and recording checkpoints: ")
			filepath= self.nw_path + "-{epoch:02d}-{loss:.4f}.hdf5"
			checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
			callbacks_list = [checkpoint]

			# fit the model
			model.fit_generator(self.TextDataGenerator(word_phrase_pairs, len_longest_phrase, n_unique_words, word_to_index, embedding), steps_per_epoch=500, epochs=50, verbose=1,callbacks=callbacks_list)
			# model.fit(X, y, epochs=50, batch_size=32, callbacks=callbacks_list)
			model.save_weights(self.nw_path + ".hdf5", overwrite=True)
			self.model = model
			loss = model.evaluate_generator(self.TextDataGenerator(word_phrase_pairs, len_longest_phrase, n_unique_words, word_to_index, embedding), 50, workers=1)
			print("LOSS: " + str(loss))
			print(model.metrics_names)

			print("[TRAINING] Done.")

		else:
			
			try:
				model.load_weights(self.nw_path)
				model.compile(loss='categorical_crossentropy', optimizer='adam')

			except Exception as e:
				print("[ERROR] " + str(e))
			self.model = model
			print("[TRAINING] Model weights loaded.")

	def sample_word_lvl(self, queue, embedding):
		w2v_model = word2vec.load('./word2vec_training/text8.bin')
		glove_model = Glove('./glove_training/all_words.glove.100.txt')
		model = self.model
		word_to_index = self.word_to_index
		index_to_word = self.index_to_word	
		X = self.X
		y = self.y	

		if embedding == 'onehot':
			queue_x = np.zeros((1, 1, self.n_unique_words))
			try: 				
				queue_x[0][0][word_to_index[queue]] = 1
			except Exception as e:
				print("[ERROR] " + str(e))
				print("[ERROR] Likely that the query is not recognized.")
				return

		elif embedding == 'word2vec':
			queue_x = np.zeros((1, 1, w2v_model.vectors.shape[1]))
			try: 				
				queue_x[0][0] = w2v_model[queue]
			except Exception as e:
				print("[ERROR] " + str(e))
				print("[ERROR] No equivalent in word2vec.")
				return
		elif embedding == 'glove':
			queue_x = np.zeros((1,1, glove_model.vectors[0].shape[0]))
			try:
				queue_x[0][0] = glove_model.model[queue]
			except Exception as e:
				print("[ERROR] " + str(e))
				print("[ERROR] No equivalent in glvoe.")
				return

		sampled_y = model.predict(queue_x, verbose=0)
		phrase = ""
		max_index = 0
		if embedding == 'onehot':
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
		
		elif embedding == 'word2vec': 
			for i in range(0, sampled_y.shape[1]):
				# kd-tree for quick nearest-neighbor lookup
				sample = sampled_y[0][i]
				# sample = sampled_y[0][i]/np.abs(sampled_y[0][i].max()) # RESCALE TO -1 to 1 from 0 to 1..
 				# sample = sample*2 - 1				
				# print(sample)
				tree = spatial.KDTree(w2v_model.vectors)							
				result = w2v_model.vectors[tree.query(sample)[1]]
				word_prediction = ""
				for w in w2v_model.vocab:
					if (w2v_model[w]==result).all() :
						# print(w)
						word_prediction = w

				if(word_prediction != "stop"):
					phrase += str(word_prediction)
					phrase += " "

		elif embedding == 'glove':
			for i in range(0, sampled_y.shape[1]):
				sample = sampled_y[0][i]
				tree = spatial.KDTree(glove_model.vectors)
				result = glove_model.vectors[tree.query(sample)[1]]
				word_prediction = ""
				for w in glove_model.vocab:
					if(glove_model.model[w]==result).all():
						word_prediction = w
				if(word_prediction != "<ENDLINE>"):
					phrase += str(word_prediction)
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


def sample_poem(query, mean=2.7):
	# topics = expand.expand(str(query), 'glove_haiku_50')	
	topics = expand.expand(str(query), 'glove_poem_pair_50', mean_level=mean)
	print("[OUTPUT] Topics: ")
	print(topics)
	q1 = topics[0]
	q2 = topics[1]
	q3 = topics[2]

	p1 = lstm_NN.sample_word_lvl(q1, embedding)
	p2 = lstm_NN.sample_word_lvl(q2, embedding)
	p3 = lstm_NN.sample_word_lvl(q3, embedding)
	# print(p1)
	# print(p2)
	# print(p3)
	topics_generated = q1 + ", " + q2 + ", " + q3
	poem = p1 + ", " + p2 + ", " + p3
	summary = query.strip() + " | " + topics_generated + " | " + poem + "\n"	

	return poem, summary

if __name__ == '__main__':

	# terminal argument parser
	ap = argparse.ArgumentParser()
	list_of_modes = ['train', 'sample', 'sample_file']
	list_of_embeddings = ['word2vec', 'onehot', 'glove']
	#ap.add_argument("-m", "--method", required=False, help="Method to use for WSD. Default = wordnet.", default="wordnet", choices = list_of_methods)
	ap.add_argument("-d", "--data", required=False, help="Training data corpus to train on.", default="all_words-wordnet.txt")
	ap.add_argument("-nw", "--network-weights", required=True, help="Filename selected for saving the network weights.")
	ap.add_argument("-m", "--mode", required=True, help="Choose between training mode or sampling mode.", default="train", choices=list_of_modes)
	ap.add_argument("-em", "--embedding", required=True, help="Choose the word embedding method.", default="onehot", choices=list_of_embeddings)
	ap.add_argument("-q", "--query", required=False, help="Query word to generaet a poem about.", default="breeze")

	args = vars(ap.parse_args())
	nw_filename = args["network_weights"]
	data_filename = args["data"] # only for testing purposes
	mode = args["mode"]
	embedding = args["embedding"]
	query = args["query"]

	#print("[SETUP] Method: " + str(method))
	print("[SETUP] Training Data: " + str(data_filename)) 
	print("[SETUP] Network Weights Filename Path: " + str(nw_filename)) 
	print("[SETUP] Word embedding: " + str(embedding))

	lstm_NN = HaikuGeneratorLSTM(data_filename, nw_filename)

	if mode == 'train':
		print("[SETUP] Training mode.")
		lstm_NN.train_word_lvl(embedding, train=True)
	elif mode == 'sample':
		print("[SETUP] Sampling mode.")
		print("[SETUP] Query word: " + str(query))
		lstm_NN.train_word_lvl(embedding, train=False)
	
		poem, summary = sample_poem(query, 2.7)
		print("[OUTPUT] Final poem: ")
		print(poem)
	elif mode == 'sample_file':
		lstm_NN.train_word_lvl(embedding, train=False)
	
		queryFileName = query
		queries = open(queryFileName, 'r').readlines()
		queries = [row.lower() for row in queries]	
		
		# create mapping of unique chars to integers
		outputFile = open("output2.7_multiinput.txt", 'a')
		for idx, query in enumerate(queries):			
						
			try:
				poem, summary = sample_poem(query, 2.7)
				outputFile.write(summary)
				print("[OUTPUT] Wrote to output file.")
			except:
				pass

			print(idx)
			if idx == 100:
				break
		outputFile.close()
