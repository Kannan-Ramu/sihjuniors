import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)
# Load the data
data = pd.read_csv('test.csv')

# Separate the features and the target variable
X = data[['Intermittent', 'Relative Humidity', 'Total Snow Depth', 'Temperature']]
y = data['label']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, predictions)

print(f'The accuracy of the model is {accuracy}')

def predic(values):
    # Convert the list of values into a DataFrame
    values_df = pd.DataFrame([values], columns=['Intermittent', 'Relative Humidity', 'Total Snow Depth', 'Temperature'])
    
    # Use the trained model to make a prediction
    prediction = model.predict(values_df)
    
    return prediction[0]


# Test the prediction function
# values = [77.62, 85.5, 80.7, 4.107]
# print(f'The predicted label for the values {values} is {predic(values)}')

# Endpoint for prediction
#Start page
@app.route('/')
def index():
    return render_template('avalanche.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.get_json("dataVal")
        print(data)
        # Assuming the input data is a list of dictionaries
        # input_data = pd.DataFrame(data)
        # print(input_data)

        print(data['dataVal'])

        # Apply threshold and create response
        pred=predic(data['dataVal'])
        print("pred  {}".format(pred))
        # results = { 'prediction': 'happen' if pred == 1 else 'not happen'}
        results = [{'avalanche': 1, 'prediction': 'happen' if pred > 0 else 'not happen'} ] 
        return jsonify({'results': results})

    except Exception as e:
        print(e)
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
