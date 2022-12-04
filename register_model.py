import joblib
import tensorflow as tf
from azureml.core import Model

def register_model(model, model_name, workspace):
    print('Starting model registration...')

    joblib.dump(model, f'{model_name}.pkl')

    reg_model = Model.register(workspace, 
        model_name=model_name,
        model_path=f'./{model_name}.pkl',
        model_framework=Model.Framework.TFKERAS, 
        model_framework_version=tf.__version__)

    print('Model is registered.')
    print('Name: ', reg_model.name, ', Version: ', reg_model.version)
    return reg_model