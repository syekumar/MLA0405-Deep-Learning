import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def gradient_descent(X, Y, iterations=2000, learning_rate=0.01):

    weight = 0
    bias = 0
    n = len(X)

    costs = []

    for i in range(iterations):

        Y_pred = weight * X + bias

        cost = mse(Y, Y_pred)
        costs.append(cost)

        dw = -(2/n) * np.sum(X * (Y - Y_pred))
        db = -(2/n) * np.sum(Y - Y_pred)

        weight -= learning_rate * dw
        bias -= learning_rate * db

        if i % 100 == 0:
            print(f"Iteration {i}: Cost={cost:.4f}")

    plt.plot(costs)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs Iterations")
    plt.show()

    return weight, bias


X = np.array([32.5,53.4,61.5,47.4,59.8,55.1,52.2,39.2,
              48.1,52.5,45.4,54.3,44.1,58.1,56.7,48.9,
              44.6,60.2,45.6,38.8])

Y = np.array([31.7,68.7,62.5,71.5,87.2,78.2,79.6,59.1,
              75.3,71.3,55.1,82.4,62.0,75.3,81.4,60.7,
              82.8,97.3,48.8,56.8])

scaler = StandardScaler()
X = scaler.fit_transform(X.reshape(-1,1)).flatten()

weight, bias = gradient_descent(X, Y)

print("Estimated Weight:", weight)
print("Estimated Bias:", bias)

Y_pred = weight * X + bias

plt.scatter(X, Y)
plt.plot(X, Y_pred, color="red")
plt.title("Linear Regression using Gradient Descent")
plt.show()
