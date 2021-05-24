from flask import Flask,render_template,request
import numpy as np
import pickle5 as pickle

app = Flask(__name__)
model = pickle.load(open('linear_regression.pkl', 'rb'))
scaler=pickle.load(open('scaler.pkl','rb'))
pca_x=pickle.load(open('pca_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.HTML')

@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    print(final_features)
    scaled_data=scaler.transform(final_features)
    print(scaled_data)
    x_pca = pca_x.transform(scaled_data)
    print(x_pca)
    prediction = model.predict(x_pca)
    print(prediction)
    output = round(prediction[0], 5)
    print(output)

    return render_template('home.html', prediction_text='Share will close on  INR {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
