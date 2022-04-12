import numpy as np
from matplotlib import pyplot as plt
import model
import prepare_data as dt
import cv2

model = model.build_model()
model.load_weights(r"D:\BaoChung\ML\Mnist_Project\best_weight.hdf5")

# plt.imshow(dt.X_test[6].reshape(28, 28), cmap='gray')
for im in dt.X_test:
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', im)
    cv2.waitKey()
    y_predict = model.predict(im.reshape(1, 28, 28, 1))
    print("Giá trị dự đoán là: ", np.argmax(y_predict))
