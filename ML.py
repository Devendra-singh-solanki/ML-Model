#importing necessary liabrary
import pickle
from flask import Flask, render_template, request

# Creating flask object 
app = Flask(__name__)
model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('standard_scaler_model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    input_features = [request.form.get('Material_Quality'), request.form.get('Additive_catalyst'), request.form.get('Plasticizer'), request.form.get('Refined_Aggregator'), request.form.get('Formulation_Duration')]
    # Scale input data using the loaded scaler
    scaled_features = scaler.transform([input_features])

    # Make a prediction using the loaded model
    prediction = model.predict(scaled_features)[0]
    output = round(prediction, 2)
    return render_template('index.html', prediction_text = f'Predicted Compression Strength is : {output}')

#Run the app. 
if __name__ == '__main__':
    app.run(debug=True)
