import pickle
from flask import Flask, render_template,request


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
# Load the scaler
scaler = pickle.load(open('scaler.pkl','rb'))

# url/
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods = ['GET','POST'])
def predict():
    input_features = [request.form.get('Material_Quality'), request.form.get('Additive_catalyst'), request.form.get('Ash_Component'), request.form.get('Water_Mix'), request.form.get('Plasticizer'), request.form.get('Moderate_Aggregator'), request.form.get('Refined_Aggregator'), request.form.get('Formulation_Duration')]
    # Scale input data using the loaded scaler
    scaled_features = scaler.transform([input_features])

    # Make a prediction using the loaded model
    prediction = model.predict(scaled_features)[0]
    output = round(prediction, 2)
    return render_template('index.html', prediction_text = f'Predicted Compression Strength of the new cement batch is : {output}')


if __name__ == '__main__':
    app.run(debug=True)
    