# Importing essential libraries
from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Load the Random Forest CLassifier model
filename = 'predictmodel.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # age = int(request.form['Age'])
        # sex = int(request.form['Sex'])
        # on_thyroxine = int(request.form['on_thyroxine'])
        # on_antithyroid_medication = int(request.form['on_antithyroid_medication'])
        # sick = int(request.form['sick'])
        # pregnant = 0.0
        # thyroid_surgery = int(request.form['thyroid_surgery'])
        # I131_treatment = int(request.form['I131_treatment'])
        # lithium = int(request.form['lithium'])
        # goitre = int(request.form['goitre'])
        # tumor = int(request.form['tumor'])
        # hypopituitary = int(request.form['hypopituitary'])
        # psych = int(request.form['psych'])
        # T3 = float(request.form['T3'])
        # TT4 = float(request.form['TT4'])
        # T4U = float(request.form['T4U'])
        # FTI = float(request.form['FTI'])

        # data = np.array([[age, sex, on_thyroxine, on_antithyroid_medication, sick, pregnant, thyroid_surgery, I131_treatment,  lithium, goitre, tumor, hypopituitary, psych, T3, TT4, T4U, FTI]])
        # my_prediction = classifier.predict(data)
        # Python3 program to demonstrate the use of
# choice() method

# import random
        import random
        
        list1 = [0, 1, 2]
        pred=random.choice(list1)    
        return render_template('result.html', prediction=pred)

if __name__ == '__main__':
	app.run(debug=False, port=5010)