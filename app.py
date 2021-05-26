from flask import Flask,render_template,request
import numpy as np
import pickle5 as pickle

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.HTML')

@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    print(features)
    if(features[0] == "None"):
        return render_template('home.HTML', prediction_text='Please select a share.')

    name = features[0]
    feat = features[1:]
    int_features = [float(x) for x in feat]
    final_features = [np.array(int_features)]
    print(final_features)

    output = 0.0
    if (name == "HDFC"):
        print("HDFC")
        scaler = pickle.load(open('model/HDFC_Scaler.pkl','rb'))
        pca_x = pickle.load(open('model/HDFC_pca.pkl','rb'))
        scaled_data = scaler.transform(final_features)
        x_pca = pca_x.transform(scaled_data)
        model = pickle.load(open('model/linear_regression.pkl', 'rb'))
        prediction = model.predict(x_pca)
        print(prediction)
        output = round(prediction[0], 5)

    elif (name == "AxisBank"):
        print("Axis")
        scaler = pickle.load(open('model/Axis_bank_Scaler.pkl','rb'))
        pca_x = pickle.load(open('model/Axis_bank_pca.pkl','rb'))
        scaled_data = scaler.transform(final_features)
        x_pca = pca_x.transform(scaled_data)
        model = pickle.load(open('model/Axixbank.pkl', 'rb'))
        prediction = model.predict(x_pca)
        print(prediction)
        output = round(prediction[0], 5)

    elif (name == "BajajAuto"):
        print("BajajAuto")
        scaler = pickle.load(open('model/Bajaj-auto_Scaler.pkl','rb'))
        pca_x = pickle.load(open('model/Bajaj-auto_pca.pkl','rb'))
        scaled_data = scaler.transform(final_features)
        x_pca = pca_x.transform(scaled_data)
        model = pickle.load(open('model/Bajaj-auto.pkl', 'rb'))
        prediction = model.predict(x_pca)
        print(prediction)
        output = round(prediction[0], 5)

    elif (name == "Britania"):
        print("Britania")
        scaler = pickle.load(open('model/Britannia_Scaler.pkl','rb'))
        pca_x = pickle.load(open('model/Britannia_pca.pkl','rb'))
        scaled_data = scaler.transform(final_features)
        x_pca = pca_x.transform(scaled_data)
        model = pickle.load(open('model/Britannia.pkl', 'rb'))
        prediction = model.predict(x_pca)
        print(prediction)
        output = round(prediction[0], 5)

    elif (name == "ICICIBank"):
        print("ICICIBank")
        scaler = pickle.load(open('model/ICICI_Scaler.pkl','rb'))
        pca_x = pickle.load(open('model/ICICI_pca.pkl','rb'))
        scaled_data = scaler.transform(final_features)
        x_pca = pca_x.transform(scaled_data)
        model = pickle.load(open('model/ICICI.pkl', 'rb'))
        prediction = model.predict(x_pca)
        print(prediction)
        output = round(prediction[0], 5)

    elif (name == "KotakBank"):
        print("KotakBank")
        scaler = pickle.load(open('model/Kotakbank_Scaler.pkl','rb'))
        pca_x = pickle.load(open('model/Kotakbank_pca.pkl','rb'))
        scaled_data = scaler.transform(final_features)
        x_pca = pca_x.transform(scaled_data)
        model = pickle.load(open('model/Kotakbank.pkl', 'rb'))
        prediction = model.predict(x_pca)
        print(prediction)
        output = round(prediction[0], 5)

    elif (name == "Reliance"):
        print("Reliance")
        scaler = pickle.load(open('model/Reliance_Scaler.pkl','rb'))
        pca_x = pickle.load(open('model/Realiance_pca.pkl','rb'))
        scaled_data = scaler.transform(final_features)
        x_pca = pca_x.transform(scaled_data)
        model = pickle.load(open('model/Reliance.pkl', 'rb'))
        prediction = model.predict(x_pca)
        print(prediction)
        output = round(prediction[0], 5)

    elif (name == "Titan"):
        print("Titan")
        scaler = pickle.load(open('model/Titan_Scaler.pkl','rb'))
        pca_x = pickle.load(open('model/Titan_pca.pkl','rb'))
        scaled_data = scaler.transform(final_features)
        x_pca = pca_x.transform(scaled_data)
        model = pickle.load(open('model/Titan.pkl', 'rb'))
        prediction = model.predict(x_pca)
        print(prediction)
        output = round(prediction[0], 5)

    elif (name == "Wipro"):
        print("Wipro")
        scaler = pickle.load(open('model/Wipro_Scaler.pkl','rb'))
        pca_x = pickle.load(open('model/Wipro_pca.pkl','rb'))
        scaled_data = scaler.transform(final_features)
        x_pca = pca_x.transform(scaled_data)
        model = pickle.load(open('model/Wipro.pkl', 'rb'))
        prediction = model.predict(x_pca)
        print(prediction)
        output = round(prediction[0], 5)

    elif (name == "Maruti"):
        print("Maruti")
        scaler = pickle.load(open('model/Maruti_Scaler.pkl','rb'))
        pca_x = pickle.load(open('model/Maruti_pca.pkl','rb'))
        scaled_data = scaler.transform(final_features)
        x_pca = pca_x.transform(scaled_data)
        model = pickle.load(open('model/Maruti.pkl', 'rb'))
        prediction = model.predict(x_pca)
        print(prediction)
        output = round(prediction[0], 5)

    return render_template('home.HTML', prediction_text='Share will close on  INR {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
