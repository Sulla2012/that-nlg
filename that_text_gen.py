import os
import tensorflow as tf
import numpy as np

def split_input_target(chunk):
	input_text = chunk[:-1]
	target_text = chunk[1:]
	return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                batch_input_shape=[batch_size,None]),
        tf.keras.layers.LSTM(rnn_units, return_sequences = True, stateful = True,
                recurrent_initializer = 'glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

class model:
	def __init__(self, name = None, seq_len = 100):
		
		if name is None:
			self.name = 'generic'
			essays = open('../../texts/essays.txt', 'rb').read().decode(encoding='utf-8')
			self.text = essays
		else:
			self.name = name
			text = open('../../texts/{}.txt'.format(self.name), 'rb').read().decode(encoding='utf-8')
			essays = open('../../texts/essays.txt', 'rb').read().decode(encoding='utf-8')

			text = text + essays
			self.text = text

		self.vocab = sorted(set(self.text))
		self.char2idx = {unique:idx for idx, unique in enumerate(self.vocab)}
		self.idx2char = np.array(self.vocab)
		self.text_as_int = np.array([self.char2idx[char] for char in self.text])
		self.char_dataset = tf.data.Dataset.from_tensor_slices(self.text_as_int)

		self.seq_len = seq_len
		self.sequences = self.char_dataset.batch(self.seq_len + 1, drop_remainder = True)
		
		self.dataset = self.sequences.map(split_input_target)
		
		self.BATCH_SIZE = 64
		self.BUFFER_SIZE = 10000
		self.vocab_size = len(self.vocab)
		self.embedding_dim = 256
		self.rnn_units = 1024
		

	def load_model(self):

		self.dataset = self.dataset.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE, drop_remainder = True)

		self.generative_model = build_model(self.vocab_size, self.embedding_dim, self.rnn_units, batch_size = 1)	

		checkpoint_path = './final_weights/{}/'.format(self.name)
		self.generative_model.load_weights(checkpoint_path)
		
		self.generative_model.build(tf.TensorShape([1,None]))

	def generate_text(self):
		i = int(np.random.randint(0, len(self.text)-100))
		start_string = self.text[i:int(i+100)]
		num_gen = int(np.random.normal(1000, 300))
		input_eval = [self.char2idx[s] for s in start_string]
		input_eval = tf.expand_dims(input_eval, 0)
    
		text_generated = []
		temperature = 1.0
		try:
			self.generative_model.reset_states()	
		except:
			print('Model not loaded')
			return
		for i in range(num_gen):
			predictions = self.generative_model(input_eval)
			predictions = tf.squeeze(predictions, 0)
			predictions = predictions / temperature
			predicted_id = tf.random.categorical(predictions, num_samples = 1)[-1,0].numpy()
        
			input_eval = tf.expand_dims([predicted_id], 0)
			text_generated.append(self.idx2char[predicted_id])
		while self.idx2char[predicted_id] != '.':
			predictions = self.generative_model(input_eval)
			predictions = tf.squeeze(predictions, 0)
			predictions = predictions / temperature
			predicted_id = tf.random.categorical(predictions, num_samples = 1)[-1,0].numpy()
        
			input_eval = tf.expand_dims([predicted_id], 0)
			text_generated.append(self.idx2char[predicted_id])
		ret_str = start_string + ''.join(text_generated)
		ret_str = ret_str.replace('\n', ' ')
		ret_str = ret_str.split('.')
		ret_str = '.'.join(ret_str[1:])
		return (ret_str)


