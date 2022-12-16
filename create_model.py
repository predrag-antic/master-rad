import tensorflow as tf

def create_model(img_size, classes, learning_rate):
    print('Model creation...')

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(img_size, img_size, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D((2,2)),
        tf.keras.layers.GlobalAvgPool2D(),
        tf.keras.layers.Dense(units=len(classes), 
                            activation="sigmoid")
    ])

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Input(shape=(img_size, img_size, 1)),
    #     tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
    #     tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'),
    #     tf.keras.layers.MaxPooling2D((2,2)),
    #     tf.keras.layers.SeparableConv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
    #     tf.keras.layers.SeparableConv2D(128, kernel_size=(3,3), activation='relu', padding='same'),
    #     tf.keras.layers.MaxPooling2D((2,2)),
    #     tf.keras.layers.SeparableConv2D(256, kernel_size=(3,3), activation='relu', padding='same'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.SeparableConv2D(256, kernel_size=(3,3), activation='relu', padding='same'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.MaxPooling2D((2,2)),
    #     tf.keras.layers.SeparableConv2D(512, kernel_size=(3,3), activation='relu', padding='same'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.SeparableConv2D(512, kernel_size=(3,3), activation='relu', padding='same'),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.SeparableConv2D(512, kernel_size=(3,3), activation='relu', padding='same'),
    #     tf.keras.layers.MaxPooling2D((2,2)),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(1024, activation="relu"),
    #     tf.keras.layers.Dropout(0.7),
    #     tf.keras.layers.Dense(512, activation="relu"),
    #     tf.keras.layers.Dropout(0.5),
    #     tf.keras.layers.Dense(units=len(classes), 
    #                         activation="sigmoid")
    # ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=1e-5),
        metrics=["accuracy"]
    )
    
    print('Model is created.')
    return model