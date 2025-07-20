import pickle

with open("processed/test_balanced.pkl", "rb") as f:
    test_data = pickle.load(f)

print(type(test_data))
print("Keys:", test_data.keys())

# Print first 2 rows/items of each value in the dict
for key, value in test_data.items():
    print(f"\nKey: {key}")
    try:
        print(value[:2])
    except Exception as e:
        print(f"Cannot slice value for key '{key}': {e}")
