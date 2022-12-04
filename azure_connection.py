import azureml.core
from azureml.core import Workspace

def connect_to_azure_workspace():
    print("Ready to use AzureML", azureml.core.VERSION)

    try:
        print("Loading existing Workspace...")
        ws = Workspace.from_config('config.json')
        print(ws.name, "is loaded.")
        return ws
    except:
        print("Loading failed.")