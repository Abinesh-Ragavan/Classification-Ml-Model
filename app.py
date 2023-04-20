from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_placement():
    g1 = float(request.form.get('g1'))
    g2 = int(request.form.get('g2'))
    g3 = float(request.form.get('g3'))
    g4 = float(request.form.get('g4'))
    g5 = float(request.form.get('g5'))
    g6 = float(request.form.get('g6'))

    result = model.predict(
        np.array([g1, g2, g3, g4, g5, g6]).reshape(1, 6))
    name = int(result)

    return render_template('predict.html', value=name)


if __name__ == '__main__':
    app.run(debug=True)
