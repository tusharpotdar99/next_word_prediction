import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import pickle

# nltk.download('punkt_tab')
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

path = 'data/emailSentences.txt'


def text():
    corpus = []
    with open(path, 'r') as file:
        sentences = file.readlines()

    for sentence in sentences:
        clean_sentence = sentence.strip().lower()
        corpus.append(clean_sentence)

    return corpus


def read_file():
    corpus = []
    with open(path, 'r') as file:
        sentences = file.readlines()

    for sentence in sentences:
        clean_sentence = sentence.strip().lower()
        corpus.append(clean_sentence)

    corpus_series = pd.Series(corpus)
    tokenized_corpus = [word_tokenize(sentence.lower()) for sentence in corpus]
    return tokenized_corpus

    # Generate sequences of n-grams (e.g., trigrams: n=3)


def generate_ngrams(token_list, n=3):
    ngrams = []
    for tokens in token_list:
        for i in range(len(tokens) - n + 1):
            ngrams.append(tokens[i:i + n])
    return ngrams


def word2vec():
    tokenized_corpus = read_file()
    word2vec_model = Word2Vec(sentences=tokenized_corpus, vector_size=50, window=3, min_count=1, workers=4)
    word2vec_model.save('models/w2v_model.model')
    return word2vec_model


def data():
    tokenized_corpus = read_file()
    ngrams = generate_ngrams(tokenized_corpus, n=3)
    word2vec_model = word2vec()
    X = []
    y = []

    for ngram in ngrams:
        X.append(ngram[:-1])
        y.append(ngram[-1])

    tokenizer = word2vec_model.wv.key_to_index
    X_encoded = [[tokenizer[word] for word in seq] for seq in X]
    X_padded = pad_sequences(X_encoded, maxlen=2, padding='pre')
    y_encoded = [tokenizer[word] for word in y]
    y_onehot = to_categorical(y_encoded, num_classes=587)
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Input shape: {X_padded.shape}, Target shape: {y_onehot.shape}")
    return X_padded, y_onehot


def model():
    X_padded, y_onehot = data()
    model_nxt = Sequential()
    model_nxt.add(Embedding(input_dim=587, output_dim=50, input_length=2))
    model_nxt.add(Bidirectional(LSTM(128)))
    model_nxt.add(Dense(587, activation='softmax'))
    model_nxt.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model_nxt.summary()

    model_nxt.fit(X_padded, y_onehot, epochs=100, batch_size=32)
    model_nxt.save('models/next_word.h5')

    return model_nxt


def predict_top_n_words_lstm(input_text, top_n=5, word=None):
    corpus = text()
    with open('models/tokenizer.pkl', 'rb') as file:
        tokenizer = pickle.load(file)
    models = load_model('models/next_word.h5')
    # word2vec_model = gensim.models.keyedvectors.load_word2vec_format('models/word2vec_model.bin', binary=True)
    w2v_model = Word2Vec.load('models/w2v_model.model')
    encoded_input = [tokenizer[word] for word in input_text.split() if word in tokenizer]
    padded_input = pad_sequences([encoded_input], maxlen=2, padding='pre')
    prediction = models.predict(padded_input, verbose=0).flatten()
    top_n_indices = prediction.argsort()[-top_n:][::-1]  # Sort in descending order
    top_n_words = [w2v_model.wv.index_to_key[i] for i in top_n_indices]
    all_suggestions = []
    all_suggestions.extend(
        [sentence for sentence in corpus if input_text.lower() in sentence.lower()]
    )
    extended_input = f"{input_text} {word}"
    all_suggestions.extend(
        [sentence for sentence in corpus if extended_input.lower() in sentence.lower()]
    )
    unique_suggestions = list(dict.fromkeys(all_suggestions))
    return top_n_words, unique_suggestions[:5]


def main():
    model_nxt = model()
    texts = "i will"
    words, sentences = predict_top_n_words_lstm(texts)
    print(f"Top 10 word predictions: {words}")
    print(f"Top 10 sentences predictions: {sentences}")


if __name__ == '__main__':
    main()
