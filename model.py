from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input


# Define model
def build_model():
    input_shape = Input((28, 28, 1), name='input')
    x = Conv2D(32, (3, 3), activation='sigmoid')(input_shape)
    x = Conv2D(32, (3, 3), activation='sigmoid')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)  # Flatten layer chuyển từ tensor sang vector
    x = Dense(128, activation='sigmoid')(x)
    x = Dense(10, activation='softmax')(x)  # Output với 10 nodes

    model = Model(input_shape, x)
    return model


if __name__ == '__main__':
    build_model()
