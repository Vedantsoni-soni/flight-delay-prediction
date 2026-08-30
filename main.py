import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Sample data bana rahe hain (baad me real CSV se replace kar denge)
data = {
    'Airline': [0,1,2,0,1,2,0,1],
    'Source': [0,1,2,0,1,2,0,1],
    'Destination': [1,2,0,1,2,0,1,2],
    'Duration': [120, 180, 240, 150, 200, 300, 130, 210],
    'Total_Stops': [0,1,1,0,2,1,0,1],
    'Delay': [0,1,1,0,1,1,0,0]
}
df = pd.DataFrame(data)

X = df.drop('Delay', axis=1)
y = df['Delay']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {acc*100}%")
print("Model training done!")

# Model save kar liya
pickle.dump(model, open('flight_model.pkl','wb'))