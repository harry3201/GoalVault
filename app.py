import gradio as gr
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "student_model_compressed.tflite")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

def predict_goal(Income, Rent, Loan_Repayment, Insurance, Groceries, Utilities, Transport,
                 Entertainment, Savings, Investments, Healthcare, Education, Miscellaneous):
    features = np.array([[Income, Rent, Loan_Repayment, Insurance, Groceries, Utilities,
                          Transport, Entertainment, Savings, Investments, Healthcare,
                          Education, Miscellaneous]], dtype=np.float32)
    
    interpreter.set_tensor(input_index, features)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)[0][0]
    return f"Goal Met ✅ (Confidence: {prediction:.2f})" if prediction > 0.5 else f"Goal Not Met ❌ (Confidence: {prediction:.2f})"

demo = gr.Interface(
    fn=predict_goal,
    inputs=[
        gr.Number(label="Income"), gr.Number(label="Rent"),
        gr.Number(label="Loan Repayment"), gr.Number(label="Insurance"),
        gr.Number(label="Groceries"), gr.Number(label="Utilities"),
        gr.Number(label="Transport"), gr.Number(label="Entertainment"),
        gr.Number(label="Savings"), gr.Number(label="Investments"),
        gr.Number(label="Healthcare"), gr.Number(label="Education"),
        gr.Number(label="Miscellaneous")
    ],
    outputs="text",
    title="GoalVault – Financial Goal Predictor",
    description="Predict if a user's financial goal will be met based on expense patterns."
)

if __name__ == "__main__":
    demo.launch()
