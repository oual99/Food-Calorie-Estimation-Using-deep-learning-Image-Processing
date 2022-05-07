from flask import Flask, flash, request, redirect, url_for, render_template, session
from prediction import *
from foodCal import *
from connectdb import *

model = models.load_model('saved_model_fine_tuning/')

app = Flask(__name__)
app.secret_key = "deepfood"

@app.route('/a')

def home1():
    return render_template('index.html')


@app.route('/aaa', methods=['GET', 'POST'])

def upload_img():
    if request.method == 'POST':
        file = request.files['file']
        name = file.filename
    
        file.save('static/'+name)

        img_path = 'static/'+name
        
        label, val = getPrediction(model, img_path)
        foodCalories, mass = calories(label,img_path)
        T1, T2 = getMacnutr(label)
        nutr = {'carb':T1[0], 'fat':T1[1], 'prot':T1[2]}
        nutrg = {'carb':round(mass*T2[0]/100,2), 'fat':round(mass*T2[1]/100,2), 'prot':round(mass*T2[2]/100,2)}

        session["name"] = name
        session["label"] = label
        session["cal"] = round(foodCalories,2)
        session["val"] = val
        session["nutr"] = nutr
        session["nutrg"] = nutrg
        session["mass"] = round(mass,2)
    return render_template("index.html", img = session["name"], fruit = session["label"], cal = session["cal"], val = session["val"], nutr = session["nutr"], nutrg = session["nutrg"], mass = session["mass"])

@app.route('/bbb', methods=['GET', 'POST'])
def upload_img1():
    if request.method =='POST':
        
        
        ####
        
        newmass = int(request.form['fmass'])
        newcal = getCalorie1(session["label"], newmass)

        T1, T2 = getMacnutr(session["label"])
        nutr = {'carb':T1[0], 'fat':T1[1], 'prot':T1[2]}
        nutrg = {'carb':round(newmass*T2[0]/100,2), 'fat':round(newmass*T2[1]/100,2), 'prot':round(newmass*T2[2]/100,2)}
    return render_template("index.html", img = session["name"], fruit = session["label"], cal = newcal, val = session["val"], nutr = nutr, nutrg = nutrg, mass = session["mass"],newmass=newmass)

if __name__ == "__main__":
    app.run(debug=True)