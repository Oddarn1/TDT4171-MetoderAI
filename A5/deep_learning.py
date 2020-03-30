import pickle

from keras import Sequential
from keras.layers import *
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def load_data(filename):
    print("Loading dataset...")
    data = pickle.load(open(filename, "rb"))

    # Too long datastrings, optimization
    max_length = data["max_length"]//20
    vocab_size = data["vocab_size"]

    print("Padding sequences...")
    # Sequences longer than max_length will be shortened.
    x_data = pad_sequences(data["x_train"], maxlen=max_length)
    x_test = pad_sequences(data["x_test"], maxlen=max_length)

    # Change to binary matrix to get more precise results
    y_data = to_categorical(data["y_train"], num_classes=2)
    y_test = to_categorical(data["y_test"], num_classes=2)

    return x_data, y_data, x_test, y_test, max_length, vocab_size


def init_model(vocab_size, max_length):
    model = Sequential()

    model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_length))
    # Computer limitations makes it necessary to have fewer nodes here
    model.add(LSTM(64))
    model.add(Dense(2, activation='sigmoid'))

    # RMSprop() is good for recurrent neural networks
    model.compile(optimizer=RMSprop(), loss='binary_crossentropy', metrics=['accuracy'])

    # Print model specifications
    model.summary()

    return model


def train(model, x_data, y_data, epochs):
    # 3 epochs because of performance
    model.fit(x_data, y_data, epochs=epochs, verbose=1, batch_size=128)
    # Save trained model
    model.save("LSTM_model.h5")


def evaluate(trained_model, x_test, y_test):
    return trained_model.evaluate(x_test, y_test)


def main():
    epochs = 3

    x_data, y_data, x_test, y_test, max_length, vocab_size = load_data("keras-data.pickle")

    model = init_model(vocab_size, max_length)

    train(model, x_data, y_data, epochs)

    loss, accuracy = evaluate(model, x_test, y_test)

    print("Accuracy:%f\nLoss:%f" % (accuracy, loss))

    # Accuracy:0.923047
    # Loss:0.186883


if __name__ == "__main__":
    main()
