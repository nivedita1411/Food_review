from flask import Flask, render_template, request
import pickle

import numpy as np
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer



from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)

filename = 'nlp_model.pkl'
clf = pickle.load(open(filename,'rb'))
cv = pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    text = request.form['text']
    data = [text]
    vect = cv.transform(data).toarray()
    my_pred = clf.predict(vect)
    # # Preprocess the user's input using your preprocessing function
    # preprocessed_text = preprocess_text(text)
    # X_new = cv.transform([preprocessed_text]).toarray()
    # # Make sentiment prediction using the model
    # prediction = model.predict(np.array([X_new]).reshape(-1, 1))[0]


    # Decide how you want to display the results
    return render_template('result.html', prediction=my_pred)

if __name__ == '__main__':
    app.run(debug=True)