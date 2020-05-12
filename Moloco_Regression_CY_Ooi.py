# Moloco Regression Question
# get_ipython().magic('matplotlib notebook')
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Features normalisation function
def feature_normalise(X):
    X_norm = X
    mu = np.mean(X)
    sigma = np.std(X)
    X_norm = (X - mu) / sigma
    return [X_norm, mu, sigma]

# Function to compute cost
def compute_cost(X, y, theta, L):
    m = len(y)
    predictions = np.matmul(X, theta) + L*sum(np.square(theta))
    sqrErrors = np.square(predictions - y)
    J = 1/(2*m)*np.sum(sqrErrors)
    return J

# Gradient descent function
def gradient_descent(X, y, theta, alpha, L, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for iter in range(num_iters):
        predictions = np.matmul(X, theta) + L*sum(np.square(theta))
        delta = (1/m)*np.matmul(predictions - y, X)
        theta = theta - alpha*delta
        J_history[iter] = compute_cost(X, y, theta, L)
    return [theta, J_history]


df_reg = pd.read_excel('Adops & Data Scientist Sample Data.xlsx', sheetname='Q2 Regression', header=None)
df_reg = df_reg.rename(columns={0: 'A', 1: 'B', 2: 'C'})

# Split for train and test sets (75% train, 25% test)
df_train = df_reg.sample(frac=0.75, random_state=0)
df_test = df_reg.drop(df_train.index)

X = df_train[['A', 'B']]
y = df_train['C']
m = len(y)

# Scale features and set them to zero mean
[X, mu, sigma] = feature_normalise(X)

# Add intercept and 3-degree polynomial features to X
X['Intercept'] = 1
X = X[['Intercept', 'A', 'B']]
X['A2'] = np.square(X['A'])
X['B2'] = np.square(X['B'])
X['AB'] = np.multiply(X['A'], X['B'])
X['A3'] = X['A']**3
X['B3'] = X['B']**3
X['A2B'] = np.multiply(np.square(X['A']), X['B'])
X['AB2'] = np.multiply(X['A'], np.square(X['B']))

# Choose the learning rate alpha and regularisation parameter L
alpha = 0.1
L = 0.001

# Initiate theta and run gradient descent
num_iters = 200
initial_theta = np.zeros(len(X.columns))
[theta, J_history] = gradient_descent(X, y, initial_theta, alpha, L, num_iters)
print('Theta computed from gradient descent = ', theta)

# Plot costs over iterations
fig = plt.figure()
plt.subplots_adjust(left=0.15)
plt.scatter(range(num_iters), J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost, J')
plt.savefig('Regression_costs_vs_iterations.jpg')

# Plot data points
x1 = df_train['A']
x2 = df_train['B']
fig2 = plt.figure()
ax = fig2.add_subplot(111, projection='3d')
ax.scatter(x1, x2, y, label='Actual values')
model = np.matmul(X, theta)
ax.scatter(x1, x2, model, label='Model predictions')
ax.set_xlabel('A')
ax.set_ylabel('B')
ax.set_zlabel('C')
ax.set_zlim(-600, 200)
ax.legend()
plt.savefig('Regression_model_plot.jpg')

# Calculate mean squared error
mse_train = (np.square(model - y)).mean()
print('Mean squared error of the train set = {:.2f}'.format(mse_train))

# Test model using the test set
X_test = df_test[['A', 'B']]
y_test = df_test['C']
[X, mu, sigma] = feature_normalise(X_test)
X['Intercept'] = 1
X = X[['Intercept', 'A', 'B']]
X['A2'] = np.square(X['A'])
X['B2'] = np.square(X['B'])
X['AB'] = np.multiply(X['A'], X['B'])
X['A3'] = X['A']**3
X['B3'] = X['B']**3
X['A2B'] = np.multiply(np.square(X['A']), X['B'])
X['AB2'] = np.multiply(X['A'], np.square(X['B']))

model_test = np.matmul(X, theta)
mse_test = (np.square(model_test - y_test)).mean()
print('Mean squared error of the test set = {:.2f}'.format(mse_test))
