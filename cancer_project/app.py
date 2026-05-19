from flask import Flask, render_template, request
from predict import predict

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_prediction():
    data = [float(x) for x in request.form.values()]
    
    result, risk, prob = predict(data)

    return render_template('index.html',
                           prediction_text=f"Result: {result}",
                           risk_text=f"Risk Level: {risk}",
                           prob_text=f"Probability: {round(prob, 2)}")

if __name__ == "__main__":
    app.run(debug=True)