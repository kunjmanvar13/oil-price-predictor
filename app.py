import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template  # Import render_template

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define feature names
categorical_features = ['Region', 'Market_Type', 'Month', 'Season', 'Subsidy_Available']
numerical_features = ['Production_Volume_Tons', 'Imports_Tons', 'Exports_Tons', 'Consumption_Tons', 
                      'Industrial_Use_Tons', 'Inflation_Rate', 'Fuel_Price_Per_Liter', 'Weather_Index']
all_features = categorical_features + numerical_features

# Initialize Flask app
app = Flask(__name__, template_folder='.')  # Specify current directory as the template folder

@app.route('/')
def index():
    return render_template('index.html')  # This will now work because render_template is imported

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request body
        data = request.get_json()
        
        # Convert numerical inputs to float
        for feature in numerical_features:
            data[feature] = float(data[feature])
        
        # Convert input to DataFrame
        input_df = pd.DataFrame([data])
        
        # Ensure correct feature order
        input_df = input_df[all_features]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Return the prediction in JSON format
        return jsonify({"predicted_price": f"{prediction:.2f}"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
