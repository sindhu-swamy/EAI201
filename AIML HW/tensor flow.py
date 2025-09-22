import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


X_train = np.linspace(0, 10, 50)
y_train = 3 * X_train + 2 + np.random.randn(*X_train.shape) * 2 

model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

model.compile(optimizer='sgd', loss='mse')
model.fit(X_train, y_train, epochs=200, verbose=0)


weights = model.layers[0].get_weights()
m_learned = weights[0][0][0]  # slope
c_learned = weights[1][0]     # intercept

print(f"Learned equation: y = {m_learned:.2f}x + {c_learned:.2f}")

plt.scatter(X_train, y_train, label="Training Data")
plt.plot(X_train, model.predict(X_train), color="red", label="Learned Line")
plt.legend()
plt.show()

