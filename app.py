from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load models
MODEL_PATH = 'models'
def load_model(filename):
    with open(os.path.join(MODEL_PATH, filename), 'rb') as f:
        return pickle.load(f)

rf_model = load_model('random_forest_model.pkl')
log_model = load_model('logistic_regression_model.pkl')
gb_model = load_model('gradient_boosting_model.pkl')

input_columns = [
    'num_passengers', 'sales_channel', 'trip_type', 'purchase_lead',
    'length_of_stay', 'flight_hour', 'flight_day', 'route',
    'booking_origin', 'wants_extra_baggage', 'wants_preferred_seat',
    'wants_in_flight_meals', 'flight_duration', 'time_of_day', 'is_weekend'
]

def preprocess(df):
    # Derive time_of_day from flight_hour
    def get_time_of_day(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'

    # Create is_weekend column from flight_day
    df['is_weekend'] = df['flight_day'].isin(['Sat', 'Sun']).astype(int)

    # Create time_of_day column
    df['time_of_day'] = df['flight_hour'].apply(get_time_of_day)

    return df

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    error_message = None

    if request.method == "POST":
        form = request.form

        try:
            data = {
                'num_passengers': int(form['num_passengers']),
                'sales_channel': form['sales_channel'],
                'trip_type': form['trip_type'],
                'purchase_lead': int(form['purchase_lead']),
                'length_of_stay': int(form['length_of_stay']),
                'flight_hour': int(form['flight_hour']),
                'flight_day': form['flight_day'],
                'route': form['route'],
                'booking_origin': form['booking_origin'],
                'wants_extra_baggage': 1 if 'extra_baggage' in form else 0,
                'wants_preferred_seat': 1 if 'preferred_seat' in form else 0,
                'wants_in_flight_meals': 1 if 'in_flight_meals' in form else 0,
                'flight_duration': float(form['flight_duration'])
            }

            input_df = pd.DataFrame([data])
            processed_df = preprocess(input_df)

            # Choose model
            model_choice = form['model_choice']
            if model_choice == 'Random Forest':
                model = rf_model
            elif model_choice == 'Gradient Boosting':
                model = gb_model
            else:
                model = log_model

            # Check for missing columns
            missing_cols = set(input_columns) - set(processed_df.columns)
            if missing_cols:
                error_message = f"❌ Error: columns are missing: {missing_cols}"
            else:
                pred = model.predict(processed_df[input_columns])[0]
                prediction = "Booking Completed ✅" if int(pred) == 1 else "Booking Not Completed ❌"

        except Exception as ex:
            error_message = f"❌ Error during prediction: {ex}"

    return render_template("index.html", prediction=prediction, error=error_message)

if __name__ == "__main__":
    app.run(debug=True)
