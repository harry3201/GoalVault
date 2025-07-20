import tensorflow as tf
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score

TFLITE_MODEL_PATH = "models/student_model_compressed.tflite"
TEST_DATA_PATH = "processed/test.pkl"

def load_test_data():
    with open(TEST_DATA_PATH, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, tuple):
        X_test, y_test = data
    elif isinstance(data, dict):
        X_test, y_test = data["X"], data["y"]
    else:
        raise ValueError("Unsupported test.pkl format")

    X_test = np.array(X_test, dtype="float32")
    y_test = np.array(y_test, dtype="float32")

    return X_test, y_test


def evaluate_tflite_model(model_path, X_test, y_test):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    predictions = []
    for i in range(len(X_test)):
        x = np.expand_dims(X_test[i], axis=0)
        interpreter.set_tensor(input_index, x)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_index)[0][0]
        predictions.append(1 if pred >= 0.5 else 0)

    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    return acc, f1

def main():
    print("Loading test data...")
    X_test, y_test = load_test_data()

    print("Evaluating compressed TFLite model...")
    acc, f1 = evaluate_tflite_model(TFLITE_MODEL_PATH, X_test, y_test)
    print(f"TFLite Test Accuracy: {acc:.4f}")
    print(f"TFLite Test F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()
