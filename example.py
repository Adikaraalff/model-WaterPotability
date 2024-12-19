import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Load Dataset
file_path = "water_potability.csv"  # Ganti sesuai dengan path file Anda
data = pd.read_csv(file_path)

# 2. Data Preprocessing
# Mengisi nilai yang hilang dengan median
data.fillna(data.median(), inplace=True)

# 3. Split Dataset
X = data.drop('Potability', axis=1)  # Fitur
y = data['Potability']  # Label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Save Model
model_filename = "model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)
print(f"Model saved as {model_filename}")

# 7. Menyimpan X_test dan y_test sebagai CSV
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

