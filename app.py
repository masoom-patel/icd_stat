# Assuming you already have a method to handle API requests and data processing
from flask import Flask, request, jsonify

app = Flask(__name__)

# Sample data structure for storing age and gender data
data = [
    {'Age': 22, 'GenderID': 'Male'},
    {'Age': 30, 'GenderID': 'Female'},
    # ... more data points
]

# Function to categorize age into buckets
def categorize_age(age):
    if age <= 25:
        return '0-25'
    elif age <= 50:
        return '25-50'
    elif age <= 65:
        return '50-65'
    elif age <= 75:
        return '65-75'
    elif age <= 85:
        return '75-85'
    elif age <= 95:
        return '85-95'
    else:
        return '95+'

# Endpoint for age and gender based insights
@app.route('/insights', methods=['GET'])
def get_insights():
    age = request.args.get('age', type=int)
    gender = request.args.get('gender', type=str)

    # Validate inputs
    if gender not in ['Male', 'Female']:
        return jsonify({'error': 'Invalid gender. Must be Male or Female.'}), 400
    
    if age is None:
        return jsonify({'error': 'Age is required.'}), 400

    # Logic to analyze the insights based on age and gender
    bucket = categorize_age(age)
    insights = [record for record in data if record['GenderID'] == gender and categorize_age(record['Age']) == bucket]
    
    return jsonify({'bucket': bucket, 'insights': insights})

# Make sure to add necessary imports and configurations