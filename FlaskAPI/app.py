import flask
from flask import Flask, jsonify, request, render_template
import json
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    #response = json.dumps({'response': 'yahhhh!'})
    df = pd.read_csv('../model_data.csv').drop(['Unnamed: 0'], axis=1)
    df['spam'] = df['label'].replace({'spam': 1, 'ham': 0})
    # need tf
    from sklearn.feature_extraction.text import TfidfVectorizer
    X_texts = df['text_clean']
    tf = TfidfVectorizer(decode_error='ignore')
    tf_texts = tf.fit(X_texts.values.astype('U'))
    # need 'word_count', 'noun_count','verb_count', 'adj_count', 'adv_count'].values
    model = load_models()

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tf.transform(data).toarray()
        word_count = len(str(message.split()))
        from textblob import TextBlob
        n_noun = check_pos_tag(message, 'noun')
        n_verb = check_pos_tag(message, 'verb')
        n_adj = check_pos_tag(message, 'adj')
        n_adv = check_pos_tag(message, 'adv')
        vect2 = np.array([word_count, n_noun, n_verb, n_adj, n_adv])
        X = np.concatenate((vect, vect2.reshape((1, 5))), axis=1)
        pred = model.predict(X)
        response = int(pred)
    return render_template('result.html', prediction=response)


def check_pos_tag(x, flag):
    pos_family = {
        'noun': ['NN', 'NNS', 'NNP', 'NNPS'],
        'pron': ['PRP', 'PRP$', 'WP', 'WP$'],
        'verb': ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'],
        'adj':  ['JJ', 'JJR', 'JJS'],
        'adv': ['RB', 'RBR', 'RBS', 'WRB']
    }
    cnt = 0
    try:
        wiki = TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt


def load_models():
    file_name = "models/model_file.p"
    with open(file_name, 'rb') as pickled:
        data = pickle.load(pickled)
        model = data['model']
    return model


if __name__ == '__main__':
    app.run(debug=True)
