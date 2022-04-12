import matplotlib.pyplot as plt
import numpy as np

import model
import prepare_data as dt
from tensorflow.python.keras.callbacks import BackupAndRestore, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Save best weight, and backup
early_stop = EarlyStopping(patience=10)
best_checkpoint = ModelCheckpoint(filepath='weights/best_weight.hdf5', verbose=1, save_best_only=True,
                                  save_weights_only=True)
backup = BackupAndRestore('backup')
callback = [best_checkpoint, early_stop, backup]

# Compile model
model = model.build_model()
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
model.summary()
# Train model
H = model.fit(dt.X_train, dt.Y_train, validation_data=(dt.X_val, dt.Y_val), batch_size=32, epochs=10, verbose=1,
              callback=callback)
# Draw pilot for loss, accuracy
fig = plt.figure()
num_of_epoch = 10
plt.plot(np.arange(0, num_of_epoch), H.history['loss'], label='training loss')
plt.plot(np.arange(0, num_of_epoch), H.history['val_loss'], label='validation loss')
plt.plot(np.arange(0, num_of_epoch), H.history['accuracy'], label='accuracy')
plt.plot(np.arange(0, num_of_epoch), H.history['val_accuracy'], label='validation accuracy')
plt.title('Accuracy and Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss|Accuracy')
plt.legend()

score = model.evaluate(dt.X_test, dt.Y_test, verbose=0)
print(score)
