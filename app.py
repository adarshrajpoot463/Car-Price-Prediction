from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the trained model
with open('car_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize the Flask app
app = Flask(__name__)

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = float(request.form['gender'])
        age = float(request.form['age'])
        salary = float(request.form['salary'])
        debt = float(request.form['debt'])
        net_worth = float(request.form['net_worth'])

        input_data = np.array([[gender, age, salary, debt, net_worth]])
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f"Estimated Car Purchase Amount: ${prediction:,.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Invalid input. Error: {e}")



if __name__ == '__main__':
    app.run(debug=True)
