from flask import Flask,request, url_for, redirect, render_template, jsonify, request
import pandas as pd
import nlp_app

# Train and build model
vectorizer = nlp_app.vectorizer_fit()
X_Train_vect = nlp_app.transform(vectorizer)
grid = nlp_app.train_nlp_model(X_Train_vect)

# Flask Setup
app = Flask(__name__)

@app.route("/")
def index():
    
    return render_template("index.html")

@app.route("/data")
def data():

    return render_template("data.html")
   
@app.route("/test")
def test():
    return render_template("test.html")

# @app.route('/predict',methods=['POST'])
# def predict():
#     int_features = [x for x in request.form.values()]
#     final = np.array(int_features)
#     data_unseen = pd.DataFrame([final], columns = cols)
#     prediction = predict_model(model, data=data_unseen, round = 0)
#     prediction = int(prediction.Label[0])
#     return render_template('home.html',pred='Expected Bill will be {}'.format(prediction))

@app.route('/predict_api', methods=['POST', 'GET'])
def predict_api():
    tweet_text = [x for x in request.form.values()]
#     final = np.array(int_features)
#     data_unseen = pd.DataFrame([final], columns = cols)
#     prediction = predict_model(model, data=data_unseen, round = 0)
#     prediction = int(prediction.Label[0])
    
    # Vectorize / transform tweet / predict
    tweet_vect = nlp_app.transform_tweet(vectorizer, tweet_text)
    prediction = nlp_app.make_prediction(grid, tweet_vect)


    return render_template("test.html", pred=prediction)


if __name__ == '__main__':
    app.run(debug=True)