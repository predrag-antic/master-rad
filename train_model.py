import tensorflow as tf
import argparse, os
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split
from mlflow.tensorflow import mlflow

from azure_connection import connect_to_azure_workspace
from conf_matrix import generate_confusion_matrix
from create_model import create_model
from register_model import register_model
from deploy_model import deploy_model

TRAIN_URL = '/Users/predrag.antic/Data/Faculty/Master/chest_xray/chest_xray/train'
TEST_URL = '/Users/predrag.antic/Data/Faculty/Master/chest_xray/chest_xray/test'
CLASSES = ["NORMAL", "PNEUMONIA"]

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', '-b', type=int, default=128)
parser.add_argument('--epochs', '-e', type=int, default=10)
parser.add_argument('--learning-rate', '-l', type=float, default=0.001)
parser.add_argument('--num-images', '-n', type=int, default=1000)
parser.add_argument('--img_size', '-i', type=int, default=128)
parser.add_argument('--train-dataset', '-trd', type=str, default=TRAIN_URL)
parser.add_argument('--test-dataset', '-ted', type=str, default=TEST_URL)
parser.add_argument('--test-split', '-ts', type=float, default=0.2)
parser.add_argument('--experiment_name', '-en', type=str, default="PneumoniaDetection")
args = parser.parse_args()

ws = connect_to_azure_workspace()

mlflow.tensorflow.autolog()
# MLflow server
# mlflow.set_tracking_uri('localhost:5000')
# AzureML server
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

mlflow.set_experiment(args.experiment_name)

def main():
    with mlflow.start_run() as run:

        mlflow.log_param("num_of_images_for_training", args.num_images)
        mlflow.log_param("image_size", args.img_size)

        classes = get_classes()
        filenames = get_filenames()
        labels = get_labels()
        boolean_labels = get_boolean_labels(labels, classes)

        X_train, X_val, y_train, y_val = get_train_test_data(filenames, boolean_labels)

        train_data = create_data_batches(X_train, y_train)
        val_data = create_data_batches(X_val, y_val, valid_data=True)

        model = create_model(args.img_size, CLASSES, args. learning_rate)
        model.summary()

        model = train_model(model, train_data, val_data)
        test_loss, test_acc = model.evaluate(val_data)

        mlflow.log_metric('test_loss', test_loss)
        mlflow.log_metric('test_accuracy', test_acc)

        predictions = get_predictions(model, val_data)
        _, val_labels = unbatchify(val_data, classes)

        preds = []
        for i in range(len(predictions)):
            preds.append(get_pred_label(predictions[i]))

        file_name = generate_confusion_matrix(val_labels, preds)
        mlflow.log_artifact(file_name)
        precision = precision_score(val_labels, preds, pos_label='NORMAL')
        recall = recall_score(val_labels, preds, pos_label='NORMAL')

        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)

        print('After model evaluation:')
        print(f'Test accuracy is {test_acc}')
        print(f'Test loss is {test_loss}')
        print(f'Precision is {precision}')
        print(f'Recall is {recall}')

        user_input = input('Do you want to register and deploy model? [y/n] ')
        if user_input.lower() == 'y':
            model_name = input('Name of the model: ')
            reg_model = register_model(model, model_name, ws)
            deploy_model(reg_model, ws)
        else: 
            print(f'Experiment {args.experiment_name} is finished.')
            
 
def get_filenames():
    return tf.io.gfile.glob(str(f'{args.train_dataset}/*/*'))

def get_classes():
    return np.array([str(tf.strings.split(item, os.path.sep)[-1].numpy())[2:-1]
                    for item in tf.io.gfile.glob(str(f'{args.train_dataset}/*'))])

def get_labels():
    normal = len(os.listdir(f'{args.train_dataset}/NORMAL'))
    pneumonia = len(os.listdir(f'{args.train_dataset}/PNEUMONIA'))
    normal_labels = ["NORMAL" for x in range(normal)]
    pneumonia_labels = ["PNEUMONIA" for x in range(pneumonia)]
    return normal_labels + pneumonia_labels

def get_boolean_labels(labels, classes):
    return [label == classes for label in labels]

def get_train_test_data(filenames, boolean_labels):
    X = filenames
    y = boolean_labels
    X,y = shuffle(X, y, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(X[:args.num_images],
                                                    y[:args.num_images],
                                                    test_size=args.test_split,
                                                    random_state=42)

    return X_train, X_val, y_train, y_val

def process_image(image_path, img_size=args.img_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[img_size, img_size])
    
    return image

def get_image_label(image_path, label):
    image = process_image(image_path)
    return image, label

def create_data_batches(X, y=None, batch_size=args.batch_size, valid_data=False, test_data=False):
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X)))
        data = data.shuffle(buffer_size=len(X)) 
        data_batch = data.map(process_image).batch(batch_size)
        return data_batch

    elif valid_data:
        print("Creating validation data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X),
                                                tf.constant(y)))
        data_batch = data.map(get_image_label).batch(batch_size)
        return data_batch

    else:
        print("Creating training data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(X),
                                                tf.constant(y)))
        data = data.shuffle(buffer_size=len(X))
        data = data.map(get_image_label)
        data_batch = data.batch(batch_size)
        return data_batch

def train_model(model, train_data, val_data):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", 
                                                    patience=5)

    print("Starting model training...")

    model.fit(x=train_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=val_data,
            validation_freq=1,
            callbacks=[early_stopping])

    print("Training finished.")

    return model

def unbatchify(data, classes):
    images = []
    labels = []

    for image,label in data.unbatch().as_numpy_iterator():
        images.append(image)
        labels.append(classes[np.argmax(label)])

    return images,labels

def get_predictions(model, val_data):
    return model.predict(val_data, verbose=1)

def get_pred_label(prediction_probabilities):
  return CLASSES[np.argmax(prediction_probabilities)]


if __name__ == "__main__":
    main()