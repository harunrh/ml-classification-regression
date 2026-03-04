import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("autompg.csv")

x = data[["weight", "displacement", "horsepower", "cylinders"]]
y = data["mpg"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_prediction = model.predict(x_test)

mean_squared_error_value = mean_squared_error(y_test, y_prediction)
r2_score_value = r2_score(y_test, y_prediction)

print(f"Mean Squared Error: {mean_squared_error_value}")
print(f"R-squared: {r2_score_value}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_prediction, color="blue")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Linear Regression Predicted vs Actual Values")
plt.show()
