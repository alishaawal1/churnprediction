from sklearn.preprocessing import StandardScaler
import joblib

# Assuming X_train is your training data
scaler = StandardScaler()
scaler.fit(X_train)  # Fit the scaler to your training data

# Save the scaler
joblib.dump(scaler, 'model/scaler.pkl')
