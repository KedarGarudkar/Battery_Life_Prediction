from flask import Flask, render_template, request
import pickle
from sklearn.preprocessing import StandardScaler
app = Flask(__name__)
model = pickle.load(open('Basic_Classifier.sav', 'rb'))
model1 = pickle.load(open('High_Regression_Log_Life.sav', 'rb'))
model2 = pickle.load(open('Low_Regression.sav', 'rb'))
@app.route('/',methods=["GET"])
def home():
    return render_template("Basic classifier.html")

standard_to = StandardScaler()

@app.route('/',methods=["POST"])
def predict():
    if request.method == 'POST':
        resp = request.form
        a = float(resp.get('min_val'))
        b = float(resp.get('log_var'))
        c = float(resp.get('dis_max'))
        d = float(resp.get('slope'))
        e = float(resp.get('chrg_t'))
        f = float(resp.get('intercept'))
        g = float(resp.get('intgrtn'))
        h = float(resp.get('irv'))
        i = float(resp.get('ird'))
        prediction = model.predict([[a,e,b,g]])
        print(prediction)
        if prediction==1:
            prediction1=model1.predict([[a, b, c, d, f, g, e, h, i]])
            output1=prediction1
            return render_template("Basic classifier.html", prediction_value="Life Cycle is HIGH {}".format(output1))
        else:
            prediction2 = model2.predict([[a, b, c, d, f, g, e, h, i]])
            output2 = prediction2
            return render_template("Basic classifier.html", prediction_value="Life Cycle is LOW {}".format(output2))
    else:
        return render_template("Basic classifier.html")


if __name__=="__main__":
    app.run(debug=True)