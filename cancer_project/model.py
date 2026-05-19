import pickle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Models
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier()
svm = SVC(probability=True)

# Train
lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

# Save models
pickle.dump(lr, open("lr.pkl", "wb"))
pickle.dump(rf, open("rf.pkl", "wb"))
pickle.dump(svm, open("svm.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("All models saved successfully!")