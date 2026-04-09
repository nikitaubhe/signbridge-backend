import json

def strip_keys(d):
    if isinstance(d, dict):
        d.pop('batch_shape', None)
        d.pop('optional', None)
        for k, v in d.items():
            strip_keys(v)
    elif isinstance(d, list):
        for item in d:
            strip_keys(item)

# Load JSON
with open('model.json', 'r') as f:
    data = json.load(f)

# Mutate removing breaking keys for TF2.10
strip_keys(data)

# Save JSON
with open('model.json', 'w') as f:
    json.dump(data, f, indent=4)

print("model.json patched successfully!")
