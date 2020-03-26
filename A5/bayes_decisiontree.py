import pickle

from sklearn import tree
from sklearn.feature_extraction.text import *
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import *


def load_data(filename):
    # Load data
    print("Loading dataset...")
    data = pickle.load(open(filename, "rb"))

    # Initialize vectorizer
    vectorizer = HashingVectorizer()

    # Extract features from training and test set
    print("Extracting features...")
    x_data = vectorizer.fit_transform(data["x_train"])
    x_test = vectorizer.fit_transform(data["x_test"])

    y_data = data["y_train"]
    y_test = data["y_test"]

    return x_data, y_data, x_test, y_test


def init(nb_or_tree):
    return BernoulliNB() if nb_or_tree == "nb" else tree.DecisionTreeClassifier(max_depth=5)


def train(model, x_data, y_data):
    print("Training...")
    model.fit(X=x_data, y=y_data)


def predict(trained_model, x_test):
    print("Predicting...")
    return trained_model.predict(x_test)


def evaluate(score, y_test):
    return accuracy_score(y_test, score)


def main(chosen_model):
    print("Running Naive Bayes") if chosen_model == "nb" else print("Running Decision Tree")
    x_data, y_data, x_test, y_test = load_data("sklearn-data.pickle")

    classifier = init(chosen_model)

    train(classifier, x_data, y_data)

    prediction = predict(classifier, x_test)

    score = evaluate(prediction, y_test)

    print("Score: %f" % score)

    # Naive Bayes: 0.798702
    # Decision tree: 0.801981


if __name__ == "__main__":
    model = input("Which model? nb/tree: ")
    main(model)
