import joblib

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


def preprocess(dataset, args, train=True):
    """

    :param dataset: pandas dataframe
    :param args: config vars dict
    :param train: if True serializes the tokenizer to use in test data, else deserializes
    :return: tuple of (x, y) array | x = (text,  set)
    """
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';', '(', ')', '[', ']', '{', '}'])

    X = dataset['text']
    X_set = dataset['set']
    y = dataset['score']

    #  preprocess - filter stopwords
    print("pre-processing input data...")

    raw_docs = X.tolist()
    processed_docs = []
    for doc in tqdm(raw_docs):
        tokens = tokenizer.tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        processed_docs.append(" ".join(filtered))

    # tokenize for keras
    print("Tokenizing input data...")
    path = args['save_folder'] + 'tokenizer.joblib'
    if train:
        tokenizer = Tokenizer(num_words=args['nb_words'], lower=True, char_level=False)
        tokenizer.fit_on_texts(processed_docs)
        word_index = tokenizer.word_index
        print("vocabulary size: ", len(word_index))

        # save tokenizer
        joblib.dump(tokenizer, path)
        print('Saved tokenizer')
    else:
        tokenizer = joblib.load(path)
        print('Restored tokenizer')


    word_seq = tokenizer.texts_to_sequences(processed_docs)
    word_seq = sequence.pad_sequences(word_seq, maxlen=args['max_seq_len'], padding='post',
                                      truncating='post')


    return [word_seq, X_set], y

