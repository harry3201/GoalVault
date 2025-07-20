import pickle
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_data():
    with open('processed/train_balanced.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('processed/test_balanced.pkl', 'rb') as f:
        test_data = pickle.load(f)
    with open('processed/teacher_preds.pkl', 'rb') as f:
        teacher_preds = pickle.load(f)
    return train_data, test_data, teacher_preds

def prepare_soft_labels(teacher_probabilities, actual_labels, alpha=0.7):
    return alpha * np.array(teacher_probabilities) + (1 - alpha) * np.array(actual_labels)

def build_student_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),  # ✅ avoids warning
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_student_model(model, X_train, y_soft, X_test, y_test):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    return model.fit(
        X_train, y_soft,
        batch_size=32, epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[early_stop], verbose=1
    )

def evaluate_model(model, X_train, y_train, X_test, y_test):
    train_proba = model.predict(X_train, verbose=0).flatten()
    test_proba = model.predict(X_test, verbose=0).flatten()
    train_preds = (train_proba > 0.5).astype(int)
    test_preds = (test_proba > 0.5).astype(int)
    return {
        'train_acc': accuracy_score(y_train, train_preds),
        'train_f1': f1_score(y_train, train_preds),
        'test_acc': accuracy_score(y_test, test_preds),
        'test_f1': f1_score(y_test, test_preds),
        'test_precision': precision_score(y_test, test_preds),
        'test_recall': recall_score(y_test, test_preds),
        'test_proba': test_proba,
        'test_preds': test_preds
    }

def main():
    print("Loading data and teacher predictions...")
    train_data, test_data, teacher_preds = load_data()
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']

    print("\nPreparing soft labels...")
    alpha = 0.7
    y_soft = prepare_soft_labels(teacher_preds['train_probabilities'], y_train.values, alpha)
    print(f"Using alpha={alpha} (teacher knowledge weight)")

    print("\nBuilding Student Model...")
    student_model = build_student_model(X_train.shape[1])
    student_model.summary()

    print("\nTraining Student Model with Knowledge Distillation...")
    train_student_model(student_model, X_train, y_soft, X_test, y_test)

    print("\nEvaluating Student Model...")
    results = evaluate_model(student_model, X_train, y_train, X_test, y_test)
    print(f"Train Accuracy: {results['train_acc']:.4f}")
    print(f"Test Accuracy: {results['test_acc']:.4f}")
    print(f"Test F1-Score: {results['test_f1']:.4f}")

    print("\nSample Predictions vs Actual:")
    sample_idx = np.random.choice(len(y_test), 5, replace=False)
    for i in sample_idx:
        print(f"Actual: {y_test.iloc[i]} | Student: {results['test_preds'][i]} ({results['test_proba'][i]:.3f})")

    os.makedirs('models', exist_ok=True)
    student_model.save('models/student_model.keras')
    with open('processed/student_preds.pkl', 'wb') as f:
        pickle.dump({'probabilities': results['test_proba'], 'predictions': results['test_preds'], 'actual': y_test.values}, f)

    print("\nFiles saved:")
    print("✓ models/student_model.keras")
    print("✓ processed/student_preds.pkl")

if __name__ == "__main__":
    main()
