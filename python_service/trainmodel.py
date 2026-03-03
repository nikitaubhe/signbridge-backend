import numpy as np
import os
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

label_map = {label: num for num, label in enumerate(actions)}

sequences, labels = [], []

print("Loading data...")

for action in actions:
    for sequence in range(no_sequences):
        window = []
        bad_sequence = False

        for frame_num in range(sequence_length):
            npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")

            res = np.load(npy_path, allow_pickle=True)
            res = np.array(res)

            # 🚫 If empty frame → skip entire sequence
            if res.shape != (63,):
                print(f"Skipping corrupted sequence: {action}/{sequence}, frame: {frame_num}, shape={res.shape}")
                bad_sequence = True
                break

            window.append(res.astype(np.float32))

        if not bad_sequence:
            sequences.append(window)
            labels.append(label_map[action])

print("\nValid sequences:", len(sequences))
print("Labels:", len(labels))

X = np.array(sequences)
y = to_categorical(labels).astype(int)

print("Final X shape:", X.shape)
print("Final y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

print("\nTraining model...\n")
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save('model.h5')

print("\nModel saved successfully!")