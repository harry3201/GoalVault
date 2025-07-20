import pickle
import numpy as np
import tensorflow as tf

# ✅ Load test data
with open("processed/test_balanced.pkl", "rb") as f:
    test_data = pickle.load(f)

X_test = test_data["X"].astype(np.float32).values
y_test = test_data["y"].values

# ✅ Load TFLite model
interpreter = tf.lite.Interpreter(model_path="models/student_model_compressed.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict_tflite(features):
    features = np.array(features, dtype=np.float32).reshape(1, -1)
    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    return float(interpreter.get_tensor(output_details[0]['index'])[0][0])

# ✅ Test first 5 samples
correct = 0
for i in range(5):
    pred = predict_tflite(X_test[i])
    predicted_label = 1 if pred >= 0.5 else 0
    print(f"Sample {i+1}: True={y_test[i]}, Pred={predicted_label}, Confidence={pred:.4f}")
    if predicted_label == y_test[i]:
        correct += 1

print(f"\nAccuracy on first 5 samples: {correct}/5")
