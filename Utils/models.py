from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Embedding, Multiply
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
import h5py
from keras.models import Model

def Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate):
    print('Creating text model....')
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim,
                        weights=[embedding_matrix],
                        input_length=seq_length,
                        trainable=False))
    model.add(LSTM(units=512, return_sequences=True, input_shape=(seq_length, embedding_dim)))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=512, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1024, activation='tanh'))
    return model

def img_model(dropout_rate):
    print('Creating image model...')
    model = Sequential()
    model.add(Dense(1024, input_dim=4096, activation='tanh'))
    return model

def vqa_model(embedding_matrix, num_words, embedding_dim, seq_length,dropout_rate, num_classes):
    vgg_model = img_model(dropout_rate)
    lstm_model = Word2VecModel(embedding_matrix, num_words, embedding_dim, seq_length, dropout_rate)
    print('Merging final model...')

    mergedOut = Multiply()([vgg_model.output, lstm_model.output])
    mergedOut = Dropout(dropout_rate)(mergedOut)
    mergedOut = Dense(1000, activation='tanh')(mergedOut)
    mergedOut = Dropout(dropout_rate)(mergedOut)
    mergedOut = Dense(num_classes, activation='softmax')(mergedOut)
    fc_model = Model([vgg_model.input, lstm_model.input], mergedOut)

    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                     metrics=['accuracy'])
    return fc_model
