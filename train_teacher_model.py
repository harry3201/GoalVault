import pickle
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_data():
    with open('processed/train_balanced.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('processed/test_balanced.pkl', 'rb') as f:
        test_data = pickle.load(f)
    return train_data, test_data

def build_teacher_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_teacher_model(model, X_train, y_train, X_test, y_test):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=50,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    return history

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
        'train_proba': train_proba,
        'train_preds': train_preds,
        'test_proba': test_proba,
        'test_preds': test_preds
    }

def main():
    print("Loading processed data...")
    train_data, test_data = load_data()
    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']

    print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")
    print("\nBuilding Teacher Model...")
    teacher_model = build_teacher_model(X_train.shape[1])
    teacher_model.summary()

    print("\nTraining Teacher Model...")
    train_teacher_model(teacher_model, X_train, y_train, X_test, y_test)

    print("\nEvaluating Teacher Model...")
    results = evaluate_model(teacher_model, X_train, y_train, X_test, y_test)
    print(f"Train Accuracy: {results['train_acc']:.4f}")
    print(f"Test Accuracy: {results['test_acc']:.4f}")
    print(f"Test F1-Score: {results['test_f1']:.4f}")

    print("\nSaving Teacher Model & Predictions...")
    os.makedirs('models', exist_ok=True)
    teacher_model.save('models/teacher_model.keras')  # ✅ updated format

    teacher_predictions = {
        'train_probabilities': results['train_proba'],
        'train_predictions': results['train_preds'],
        'train_actual': y_train.values,
        'test_probabilities': results['test_proba'],
        'test_predictions': results['test_preds'],
        'test_actual': y_test.values
    }
    with open('processed/teacher_preds.pkl', 'wb') as f:
        pickle.dump(teacher_predictions, f)

    print("✓ models/teacher_model.keras")
    print("✓ processed/teacher_preds.pkl")

if __name__ == "__main__":
    main()
