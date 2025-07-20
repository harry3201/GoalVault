import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_process_data():
    df = pd.read_csv('data/indian_finance.csv')
    
    columns_to_keep = [
        'Income', 'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 
        'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 
        'Healthcare', 'Education', 'Miscellaneous', 'Desired_Savings', 
        'Disposable_Income'
    ]
    
    df = df[columns_to_keep].copy()
    
    expense_columns = [
        'Rent', 'Loan_Repayment', 'Insurance', 'Groceries', 
        'Transport', 'Eating_Out', 'Entertainment', 'Utilities', 
        'Healthcare', 'Education', 'Miscellaneous'
    ]
    
    df['Total_Expenses'] = df[expense_columns].sum(axis=1)
    df['goal_met'] = (df['Disposable_Income'] >= df['Desired_Savings'] * 1.2).astype(int)
    
    df = df.drop(['Desired_Savings', 'Disposable_Income'], axis=1)
    
    X = df.drop('goal_met', axis=1)
    y = df['goal_met']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    os.makedirs('processed', exist_ok=True)
    
    train_data = {'X': X_train, 'y': y_train, 'scaler': scaler, 'feature_names': X.columns.tolist()}
    test_data = {'X': X_test, 'y': y_test}
    
    with open('processed/train_balanced.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    
    with open('processed/test_balanced.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    return X_train, X_test, y_train, y_test, df

def main():
    X_train, X_test, y_train, y_test, original_df = load_and_process_data()
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Total samples: {len(original_df)}")
    
    goal_dist = original_df['goal_met'].value_counts()
    print(f"Goal not met: {goal_dist[0]} ({goal_dist[0]/len(original_df)*100:.1f}%)")
    print(f"Goal met: {goal_dist[1]} ({goal_dist[1]/len(original_df)*100:.1f}%)")
    
    print("\nProcessed data sample:")
    display_df = pd.concat([X_train.head(), y_train.head()], axis=1)
    print(display_df)
    
    print("\nData saved to processed/train_balanced.pkl and processed/test_balanced.pkl")

if __name__ == "__main__":
    main()