from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

file_name = 'conf_matrix.png'

def generate_confusion_matrix(val_labels, predictions):
    cm = confusion_matrix(val_labels, predictions, labels=['NORMAL', 'PNEUMONIA'])
    plt.figure()
    plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
    plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
    plt.savefig(file_name)

    return file_name