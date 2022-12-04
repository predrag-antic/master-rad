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

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        metrics=["accuracy"]
    )
    
    print('Model is created.')
    return model