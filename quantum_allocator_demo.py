
# Quantum Resource Allocator — Neural Net Policy Model (TensorFlow)
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

# 1. Task and Node Simulation
task_names = ["QFT", "Grover", "QKD", "TopoGate", "ErrorCorr", "AIInference"]
noise_sensitivity = np.array([0.8, 0.6, 0.3, 0.1, 0.7, 0.4])  # 6 tasks
degraded_node_noise_profiles = [0.3, 0.95, 0.9]  # Simulate Node B failure

# 2. Generate Training Data
X = []
y = []

for t_ns in noise_sensitivity:
    for n_noise in degraded_node_noise_profiles:
        for _ in range(50):
            noise_input = [t_ns + np.random.normal(0, 0.02),
                           n_noise + np.random.normal(0, 0.02)]
            label = 0 if noise_input[0] <= 0.5 else 1  # decision rule
            X.append(noise_input)
            y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int32)

# 3. Define the Neural Network
model = Sequential([
    Dense(16, activation='relu', input_shape=(2,)),
    Dense(16, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.01),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# 4. Train the Model
history = model.fit(X, y, epochs=100, verbose=0)

# 5. Visualize Training Loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label="Training Loss", linewidth=2)
plt.title("Neural Net Policy Model Training")
plt.xlabel("Epoch")
plt.ylabel("Cross-Entropy Loss")
plt.grid(True)
plt.legend()
plt.show()

# 6. Evaluate Predictions
test_inputs = np.array([[0.3, 0.3], [0.7, 0.9], [0.2, 0.95]], dtype=np.float32)
preds = model.predict(test_inputs)
for i, p in enumerate(preds):
    print(f"Task-Node: {test_inputs[i]} → Prediction: {'Topological' if np.argmax(p)==0 else 'Non-Topological'} (Prob: {p[np.argmax(p)]:.2f})")
