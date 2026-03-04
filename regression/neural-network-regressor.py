import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

data = pd.read_csv("autompg.csv")

X = data[[ "weight", "displacement", "horsepower", "cylinders" ]]
y = data["mpg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Input(shape=((X_train.shape[1],))),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")

training_history = model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=0)

y_prediction = model.predict(X_test).flatten()

mse = mean_squared_error(y_test, y_prediction)
r2 = r2_score(y_test, y_prediction)

print(f"Mean Squared Error: {mse}, R-Squared Score: {r2}")

plt.figure(figsize=((8, 6)))
plt.scatter(y_test, y_prediction, color="blue", label="Predicted values")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Ideal Fit (y=x)")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Neural Network Regressor: Predicted vs Actual Values")
plt.show()
