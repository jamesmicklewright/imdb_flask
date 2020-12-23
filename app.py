from flask import Flask, jsonify, request
import pickle
import sys
import nlp_utils
import os

app = Flask(__name__)
tfidf_vect = pickle.load(open(os.path.join('.','tf_idf_review_vectorizer.sav'),'rb'))
model      = pickle.load(open(os.path.join('.','imdb_reviews_model.sav'), 'rb'))

@app.route('/')
def home():
    return jsonify(data='Welcome Home')

@app.route('/review_sentiment')
def review_sentiment():
    review = request.args.get('review','unknown')
    cleaned_review = nlp_utils.process_text(review)
    vectorized_review = tfidf_vect.transform([cleaned_review]).toarray()
    pred = model.predict_proba(vectorized_review)[0, 1]
    print(pred)
    if review == 'unknown':
        return jsonify(response='Try again and check your spelling pal')
    if pred > 0.70:
        return jsonify(response='Wow, you sure are feeling positive about that!', review = review)
    elif pred > 0.5 and pred < 0.7:
        return jsonify(response='Sounds like you liked this one', review = review)
    elif pred < 0.5 and pred > 0.2:
        return jsonify(response='Sounds like this one was not your cup of tea',review = review)
    else:
        return jsonify(response="Ok, it seems like you're really not that keen on this one..", review = review)

if __name__ == '__main__':
    app.run(host='0.0.0.0')