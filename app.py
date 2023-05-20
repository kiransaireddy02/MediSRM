from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import joblib
import pickle
import sklearn


app = Flask(__name__)


@app.route('/')
def hello_world():
	return render_template('homepage.html')


@app.route('/breastcancer')
def breast_cancer():
    return render_template('breastcancer.html')


@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')


@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')

@app.route('/resulth',methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        nmv = float(request.form['nmv'])
        tcp = float(request.form['tcp'])
        eia = float(request.form['eia'])
        thal = float(request.form['thal'])
        op = float(request.form['op'])
        mhra = float(request.form['mhra'])
        age = float(request.form['age'])
        print(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
    
    with open('models\hdisease.pkl', 'rb') as file:
        model = pickle.load(file)
    pred = model.predict(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
    return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=1, gender=gender)

@app.route('/resultd', methods=['POST'])
def resultd():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        pregnancies = request.form['pregnancies']
        glucose = request.form['glucose']
        bloodpressure = request.form['bloodpressure']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        diabetespedigree = request.form['diabetespedigree']
        age = request.form['age']
        skinthickness = request.form['skin']
        
        model = joblib.load('models\diabetes_model.sav')
        pred = model.predict([[insulin, bmi, diabetespedigree, age]])
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultd.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

@app.route('/resultbc', methods=['POST'])
def resultbc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        cpm = request.form['concave_points_mean']
        am = request.form['area_mean']
        rm = request.form['radius_mean']
        pm = request.form['perimeter_mean']
        cm = request.form['concavity_mean']
        
        model = joblib.load('models\Breast_model.sav')
        pred = model.predict(np.array([cpm, am, rm, pm, cm]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Breast Cancer test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resultbc.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


if __name__ == '__main__':
	app.run()

