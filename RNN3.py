import os
import json
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation, Embedding

data_directory = "E:/College/Final Proj/Data/"
data_file = "Data_Tunes.txt"
charIndex_json = "char_to_index.json"
model_weights_directory = 'E:/College/Final Proj/Data/Model_Weights/'


def make_model(unique_chars):
    model = Sequential()

    model.add(Embedding(input_dim=unique_chars, output_dim=512, batch_input_shape=(1, 1)))

    model.add(LSTM(256, return_sequences=True, stateful=True))
    model.add(Dropout(0.2))

    model.add(LSTM(256, return_sequences=True, stateful=True))
    model.add(Dropout(0.2))

    model.add(LSTM(256, stateful=True))

    model.add(Dropout(0.2))

    model.add((Dense(unique_chars)))
    model.add(Activation("softmax"))

    return model


def generate_sequence(epoch_num, initial_index, seq_length):
    with open(os.path.join(data_directory, charIndex_json)) as f:
        char_to_index = json.load(f)
    index_to_char = {i: ch for ch, i in char_to_index.items()}
    unique_chars = len(index_to_char)

    model = make_model(unique_chars)
    model.load_weights(model_weights_directory + "Weights_{}.h5".format(epoch_num))

    sequence_index = [initial_index]

    for _ in range(seq_length):
        batch = np.zeros((1, 1))
        batch[0, 0] = sequence_index[-1]

        predicted_probs = model.predict_on_batch(batch).ravel()
        sample = np.random.choice(range(unique_chars), size=1, p=predicted_probs)

        sequence_index.append(sample[0])

    seq = ''.join(index_to_char[c] for c in sequence_index)

    cnt = 0
    for i in seq:
        cnt += 1
        if i == "\n":
            break
    seq1 = seq[cnt:]

    cnt = 0
    for i in seq1:
        cnt += 1
        if i == "\n" and seq1[cnt] == "\n":
            break
    seq2 = seq1[:cnt]

    return seq2

ep = 90
ar = 45
ln= 400
music = generate_sequence(ep, ar, ln)

print("\nMUSIC SEQUENCE GENERATED: \n")

print(music)

