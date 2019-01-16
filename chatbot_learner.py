import numpy as np
import ast
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Input
from keras.utils import np_utils
from keras.models import Model
import codecs

file = open("cornell movie-dialogs corpus/movie_lines.txt", 'r')
file2 = open("cornell movie-dialogs corpus/movie_lines_modified.txt", 'w')

for line in file:
    file2.write(line.replace(" +++$+++ ", "@"))
file.close()
file2.close()

file = open("cornell movie-dialogs corpus/movie_conversations.txt", 'r')
file2 = open("cornell movie-dialogs corpus/movie_conversations_modified.txt", 'w')

for line in file:
    file2.write(line.replace(" +++$+++ ", "@"))
file.close()
file2.close()

filecp = codecs.open("cornell movie-dialogs corpus/movie_lines_modified.txt", encoding = 'cp1252')
movie_conversations = np.loadtxt("cornell movie-dialogs corpus/movie_conversations_modified.txt", dtype=str, delimiter="@", usecols=3)
movie_lines = np.loadtxt(filecp, dtype=str, delimiter="@",  comments="XXXc$o$m$m$e$n$tXXX", usecols=(0,4))

# movie_lines: lineID, text
# movie_conversations = order of the utterances: ['lineID1','lineID2','lineIDN']

# Create dictionary for movie lines
movie_lines_dict = {}
for i in range(len(movie_lines)):
    movie_lines_dict[movie_lines[i][0]] = movie_lines[i][1]

print("Dictionary Complete!")
movie_conversation_matrix = []
for i in range(len(movie_conversations)):
    movie_conversation_matrix.append(ast.literal_eval(movie_conversations[i]))
print("Done making movie_conversation_matrix!")

X = [] # [diaglogue1, dialogue2, ..., dialogueN]
Y = []
for i in range(len(movie_conversation_matrix)):
    for j in range(len(movie_conversation_matrix[i]) - 1):
        p1 = 0 # index of first line#
        p2 = 1 # index of second line#
        X.append(movie_lines_dict[movie_conversation_matrix[i][j]])
        Y.append(movie_lines_dict[movie_conversation_matrix[i][j+1]])

input_characters = set()
target_characters = set()
for i in range(len(X)):
    input_text = X[i]
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
            
for i in range(len(Y)):
    output_text = Y[i]
    for char in output_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in X])
max_decoder_seq_length = max([len(txt) for txt in Y])


print('Number of samples:', len(X))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)


input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(X), max_encoder_seq_length, num_encoder_tokens),
    dtype='float16')
decoder_input_data = np.zeros(
    (len(X), max_decoder_seq_length, num_decoder_tokens),
    dtype='float16')
decoder_target_data = np.zeros(
    (len(X), max_decoder_seq_length, num_decoder_tokens),
    dtype='float16')

for i, (input_text, target_text) in enumerate(zip(X, Y)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.


batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256 # Latent dimensionality of the encoding space.


# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')