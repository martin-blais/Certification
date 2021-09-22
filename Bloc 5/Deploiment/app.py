
from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('iris_model.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['Ls']
    data2 = request.form['ls']
    data3 = request.form['Lp']
    data4 = request.form['lp']
    arr = np.array([[data1, data2, data3, data4]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)

