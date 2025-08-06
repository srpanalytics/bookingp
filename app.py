from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import pickle

app = Flask(__name__)

MODEL_PATH = 'models'
INPUT_COLUMNS = [
    'num_passengers', 'sales_channel', 'trip_type', 'purchase_lead',
    'length_of_stay', 'flight_hour', 'flight_day', 'route',
    'booking_origin', 'wants_extra_baggage', 'wants_preferred_seat',
    'wants_in_flight_meals', 'flight_duration', 'is_weekend', 'time_of_day'
]

# On-demand model loading
def load_model(filename):
    with open(os.path.join(MODEL_PATH, filename), 'rb') as f:
        return pickle.load(f)

# Preprocess logic — update as per your actual pipeline
def preprocess(df):
    # Basic transformations for 'is_weekend' and 'time_of_day'
    df['is_weekend'] = df['flight_day'].apply(lambda x: 1 if x in ['Sat', 'Sun'] else 0)
    df['time_of_day'] = df['flight_hour'].apply(lambda x: (
        'Morning' if 5 <= x < 12 else
        'Afternoon' if 12 <= x < 17 else
        'Evening' if 17 <= x < 21 else
        'Night'
    ))
    return df

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None

    if request.method == "POST":
        try:
            form = request.form

            # Collect form data
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
                'flight_duration': float(form['flight_duration']),
            }

            input_df = pd.DataFrame([data])
            processed_df = preprocess(input_df)

            # Load model based on user selection
            model_choice = form['model_choice']
            if model_choice == 'Random Forest':
                model = load_model('random_forest_model.pkl')
            elif model_choice == 'Gradient Boosting':
                model = load_model('gradient_boosting_model.pkl')
            else:
                model = load_model('logistic_regression_model.pkl')

            # Predict
            pred = model.predict(processed_df)[0]
            prediction = "✅ Booking Completed" if int(pred) == 1 else "❌ Booking Not Completed"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("index.html", prediction=prediction)



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)






# if __name__ == "__main__":
#     app.run(debug=True)
