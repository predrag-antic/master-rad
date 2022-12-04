# from azureml.core.webservice import AksWebservice
# from azureml.core.compute import AksCompute
from azureml.core import Environment, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice

def deploy_model(model, ws):
    print('Starting model deployment...')

    inference_env = Environment.from_conda_specification(
        name='inference_env',
        file_path='inference_env.yml')

    inference_config = InferenceConfig(
        entry_script='entry_script.py', 
        environment=inference_env)

    aci_config = AciWebservice.deploy_configuration(
        cpu_cores=1, memory_gb=1, 
        enable_app_insights=True, 
        collect_model_data=True)

    # aks_compute = AksCompute(ws, 'aks-cluster')
    # aks_config = AksWebservice.deploy_configuration(
    #      cpu_cores=1, memory_gb=1, gpu_cores=0.1,
    #      enable_app_insights=True, 
    #      collect_model_data=True)

    service = Model.deploy(
        workspace=ws,
        name='pneumonia-detection-service',
        models=[model],
        inference_config=inference_config,
        # deployment_target=aks_compute,
        deployment_config=aci_config,
        overwrite=True,
        show_output=True)

    service.wait_for_deployment(True)
    print(service.state)
    print(service.get_logs())

    uri = service.scoring_uri
    print('Model is deployed.')
    print(f'The URI of the endpoint is: {uri}')