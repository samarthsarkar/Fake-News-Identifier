from flask import Flask, render_template, request                       # importing the libraries we need for program
from sklearn.feature_extraction.text import TfidfVectorizer             # importing the microframework flask
import pickle                                                           # importing pickle module
import pandas as pd
from sklearn.model_selection import train_test_split

app = Flask(__name__)                                                   # taking name of current module as argument
tvidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)    # running the ML train-test program program
loaded_model = pickle.load(open('model.pkl', 'rb'))
dataframe = pd.read_csv('news.csv')
x = dataframe['text']
y = dataframe['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)


def fake_news_det(news):                                                # Running program to process input
    tfidf_train = tvidf_vectorizer.fit_transform(x_train)
    tfidf_test = tvidf_vectorizer.transform(x_test)
    input_data = [news]
    vectorized_input_data = tvidf_vectorizer.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction


@app.route('/')                                                         # using route decorator to set trigger URL
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        pred = fake_news_det(message)
        print(pred)
        return render_template('index.html', prediction=pred)
    else:
        return render_template('index.html', prediction="Something went wrong")


if __name__ == '__main__':                                              # Now the code will only run if it was run
    app.run(debug=True)                                                 # directly and not as an imported module
