import tensorflow as tf
import numpy as np

m = 8.0
c = 12.0
X = np.linspace(0, 15, 200)
y = m * X + c + np.random.normal(0, 3, size=X.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])


model.compile(optimizer='adam', loss='mse')

history = model.fit(X, y, epochs=100, verbose=0)


w, b = model.layers[0].get_weights()
print(f"True slope: {m:.2f}, True intercept: {c:.2f}")
print(f"Learned slope: {w[0][0]:.2f}, Learned intercept: {b[0]:.2f}")

x_test = np.array([[20.0]])
y_pred = model.predict(x_test)
print(f"Prediction for x={x_test[0][0]} → y={y_pred[0][0]:.2f}")
