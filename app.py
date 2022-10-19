import numpy as np
import pandas as pd
from flask import Flask, request,render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    features = [float(x) for x in request.form.values()]
    test_df = pd.DataFrame([features], columns=['Age', 'Balance', 'IsActiveMember'])
    # final_features = [np.array(int_features)]
    # age = int(request.args.get("Age"))
    # balance = float(request.args.get("Balance"))
    # isActiveMember = int(request.args.get("IsActiveMember"))
    # test_df = pd.DataFrame({'Age': [age], 'Balance': [balance], 'IsActiveMember': [isActiveMember]})

    ncolumns = ['Age', 'Balance']
    test_df[ncolumns] = scaler.fit_transform(test_df[ncolumns])

    prediction = model.predict(test_df)

    prediction_text = "This person will sign up for a checking account" if prediction == 1 else "This person will not sign up for a checking account"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)