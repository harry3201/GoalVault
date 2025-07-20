# GoalVault: Financial Goal Achievement Prediction

## Project Overview

GoalVault predicts whether an individual is likely to achieve their financial goals based on their income and spending behavior.

The system is built using a **Teacher-Student Knowledge Distillation (KD)** approach:

- A **Teacher Model** (large, accurate neural network) is first trained on detailed financial data
- A smaller **Student Model** learns from the teacher, achieving similar accuracy while being significantly compressed for fast, lightweight deployment
- The final optimized model is exported as a **TensorFlow Lite (TFLite)** model and deployed via a web interface

## Problem Definition

Financial institutions and personal finance apps often need to assess financial goal feasibility based on historical spending and saving patterns. Traditional models are either too large for real-time mobile deployment or lack efficiency.

**GoalVault solves this by:**
- Using normalized income-expense features to learn spending patterns
- Offering fast predictions on resource-constrained devices after model compression

## Dataset & Features

The dataset (`indian_finance.csv`) contains normalized financial data per individual.

**Features (13):**
1. Income
2. Rent
3. Loan Repayment
4. Insurance
5. Utilities
6. Groceries
7. Transport
8. Entertainment
9. Savings
10. Healthcare
11. Education
12. Miscellaneous
13. Total Expenses


**Dataset Preview:**
<img width="2161" height="121" alt="image" src="https://github.com/user-attachments/assets/2c940d3b-81d5-494e-8367-9bc84edcb5fd" />

**Target Variable:**  
`goal_met` → `1` (goal achieved) or `0` (goal not achieved)

The data was **balanced** to avoid bias, ensuring equal representation of both classes.

## Model Training

- **Teacher Model:** A dense neural network trained on normalized financial features for maximum accuracy
- **Student Model:** Trained using **Knowledge Distillation (KD)** — learning from teacher predictions rather than raw labels
- **Compression:** Converted to **TFLite** with post-training optimizations for lightweight deployment

## Results

| Model | Accuracy | F1-Score | Size |
|-------|----------|----------|------|
| **Teacher (Keras)** | ~92% | ~95% | ~3.5 MB |
| **Student (Keras)** | ~91% | ~94% | ~64 KB |
| **Student (Compressed TFLite)** | ~90.5% | ~95% | ~8 KB |

The compressed model provides almost the same performance as the teacher while being **~400x smaller**.

## Real-World Applications

- **Personal Finance Apps:** Predict if users are on track to achieve savings or investment goals
- **Credit Risk Assessment:** Estimate financial stability based on spending patterns
- **Wealth Management Tools:** Identify clients needing personalized financial advice
- **Embedded Systems:** Run on mobile or edge devices where storage and compute are limited

## Tech Stack

- **Machine Learning:** TensorFlow, Keras, Scikit-learn
- **Data Handling:** Pandas, NumPy
- **Model Compression:** TensorFlow Lite Converter
- **Deployment:** Gradio (local & Hugging Face Spaces)
- **Visualization & Metrics:** Matplotlib, Scikit-learn metrics

## Project Structure

```
GoalVault/
├── data/
│   └── indian_finance.csv             # Raw dataset
├── models/
│   ├── teacher_model.keras            # Teacher model
│   ├── student_model.keras            # Student model (pre-compression)
│   └── student_model_compressed.tflite # Compressed final model
├── processed/                         # Preprocessed data & predictions
│   ├── train.pkl / test.pkl
│   ├── train_balanced.pkl / test_balanced.pkl
│   ├── teacher_preds.pkl / student_preds.pkl
├── app_gradio.py                      # Web app for predictions
├── train_teacher_model.py             # Teacher model training
├── train_student_model.py             # Student model training (KD)
├── compress_and_export_student.py     # Model compression script
├── evaluate_compressed_model.py       # TFLite evaluation
└── requirements.txt                   # Dependencies
```

## Installation & Running Locally

### Step 1: Clone the Repository
```bash
git clone https://huggingface.co/spaces/TensionFlow78/GoalVault
cd GoalVault
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch the Gradio App
```bash
python app_gradio.py
```

The Basic interface will be available at: **http://127.0.0.1:7860/**


It will look like :
<img width="1884" height="881" alt="image" src="https://github.com/user-attachments/assets/98be2e9d-f81b-4bb9-8454-e19418b516db" />


## Live Demo

You can access the deployed version here:  
**[Hugging Face Space – GoalVault](https://huggingface.co/spaces/TensionFlow78/GoalVault)**

## Future Work

- Ablation studies on different KD settings to optimize compression further
- Adding explainability (**SHAP/LIME**) to provide reasoning for predictions
- Expanding to **multi-goal prediction** (e.g., retirement vs emergency savings)

## License

Open-sourced under the **MIT License**.
