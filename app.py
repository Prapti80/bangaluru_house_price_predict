from flask import Flask, render_template, request
import pickle
import json
import pandas as pd

app = Flask(__name__)

# Load trained model
with open('bangalore_home_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load columns (for dropdown)
with open('bangalore_home_price_model_columns.json', 'r') as f:
    columns_data = json.load(f)

locations = [loc.replace("location_", "") for loc in columns_data['location_columns']]


def predict_price(location, sqft, bath, bhk):
    input_df = pd.DataFrame([{
        "total_sqft": sqft,
        "bath": bath,
        "bhk": bhk,
        "location": location
    }])

    return round(model.predict(input_df)[0], 2)


@app.route('/')
def home():
    return render_template('index.html', locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    sqft = float(request.form['sqft'])
    bath = int(request.form['bath'])
    bhk = int(request.form['bhk'])
    location = request.form['location']

    price = predict_price(location, sqft, bath, bhk)

    return render_template(
        'index.html',
        prediction_text=f"Estimated Price: ₹ {price} Lakhs",
        locations=locations
    )


if __name__ == "__main__":
    app.run(debug=True)
