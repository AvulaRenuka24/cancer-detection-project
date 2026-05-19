import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load models
lr = pickle.load(open("lr.pkl", "rb"))
rf = pickle.load(open("rf.pkl", "rb"))
svm = pickle.load(open("svm.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

def risk_score(prob):
    if prob > 0.7:
        return "HIGH RISK"
    elif prob > 0.4:
        return "MEDIUM RISK"
    else:
        return "LOW RISK"

def plot_graph(lr_prob, rf_prob, svm_prob, avg_prob):
    models = ['LR', 'RF', 'SVM', 'Hybrid']
    values = [lr_prob, rf_prob, svm_prob, avg_prob]

    plt.figure()
    plt.bar(models, values)
    plt.title("Model Comparison")
    plt.ylabel("Probability")
    plt.savefig("static/graph.png")
    plt.close()

def predict(data):
    data = np.array(data).reshape(1, -1)
    data = scaler.transform(data)

    lr_prob = lr.predict_proba(data)[0][1]
    rf_prob = rf.predict_proba(data)[0][1]
    svm_prob = svm.predict_proba(data)[0][1]

    avg_prob = (lr_prob + rf_prob + svm_prob) / 3

    plot_graph(lr_prob, rf_prob, svm_prob, avg_prob)

    result = "BENIGN" if avg_prob > 0.5 else "MALIGNANT"
    risk = risk_score(avg_prob)

    return result, risk, avg_prob