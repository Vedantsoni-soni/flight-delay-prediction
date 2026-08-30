import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Dummy realistic data banate hain (tumhare paas real CSV hai to yahan load kar dena)
data = {
    'Airline': ['IndiGo', 'Air India', 'SpiceJet', 'Vistara']*250,
    'Source': ['DEL', 'BOM', 'BLR', 'HYD']*250,
    'Destination': ['BOM', 'DEL', 'HYD', 'BLR']*250,
    'Stops': [0, 1, 0, 1]*250,
    'Dep_Hour': list(np.random.randint(0, 23, 1000)),
    'Delay': list(np.random.randint(5, 90, 1000))
}
df = pd.DataFrame(data)

# Rule lagate hain taaki accurate lage
df.loc[df['Airline'] == 'SpiceJet', 'Delay'] += 20
df.loc[df['Dep_Hour'] > 20, 'Delay'] += 15 # raat ko zyada delay

le_air = LabelEncoder()
le_src = LabelEncoder()
le_dest = LabelEncoder()

df['Airline'] = le_air.fit_transform(df['Airline'])
df['Source'] = le_src.fit_transform(df['Source'])
df['Destination'] = le_dest.fit_transform(df['Destination'])

X = df[['Airline', 'Source', 'Destination', 'Stops', 'Dep_Hour']]
y = df['Delay']

model = RandomForestRegressor()
model.fit(X, y)

# Save kar do
pickle.dump(model, open('flight_model.pkl', 'wb'))
pickle.dump(le_air, open('le_air.pkl', 'wb'))
pickle.dump(le_src, open('le_src.pkl', 'wb'))
pickle.dump(le_dest, open('le_dest.pkl', 'wb'))

print("Model ready! flight_model.pkl ban gaya")