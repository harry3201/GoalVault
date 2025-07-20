import tensorflow as tf
import os

MODEL_PATH = "models/student_model.keras"
OUTPUT_PATH = "models/student_model_compressed.tflite"

def compress_and_export_student(model_path, output_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    print("Loading student model...")
    model = tf.keras.models.load_model(model_path)

    print("Converting model to TFLite with dynamic range quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    original_size = os.path.getsize(model_path) / 1024
    compressed_size = os.path.getsize(output_path) / 1024

    print(f"Original model size: {original_size:.2f} KB")
    print(f"Compressed model size: {compressed_size:.2f} KB")
    print(f"Saved compressed model to {output_path}")

if __name__ == "__main__":
    compress_and_export_student(MODEL_PATH, OUTPUT_PATH)
