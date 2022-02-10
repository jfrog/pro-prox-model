import requests

resp = requests.request(
    url="https://valohai-prod-is.jfrog.org/api/v0/pipelines/",
    method="POST",
    headers={"Authorization": "Token YOUR_TOKEN_HERE"},
    json={
        "edges": [
            {
                "source_node": "load_data_train",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "process_train",
                "target_type": "input",
                "target_key": "loaded_data"
            },
            {
                "source_node": "process_train",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "fit_rf",
                "target_type": "input",
                "target_key": "processed_data"
            },
            {
                "source_node": "process_train",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "fit_lgb",
                "target_type": "input",
                "target_key": "processed_data"
            },
            {
                "source_node": "process_train",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "fit_cbc",
                "target_type": "input",
                "target_key": "processed_data"
            },
            {
                "source_node": "process_train",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "fit_hist",
                "target_type": "input",
                "target_key": "processed_data"
            },
            {
                "source_node": "fit_rf",
                "source_key": "*.sav",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "rf"
            },
            {
                "source_node": "fit_lgb",
                "source_key": "*.sav",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "lgb"
            },
            {
                "source_node": "fit_cbc",
                "source_key": "*.sav",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "cbc"
            },
            {
                "source_node": "fit_hist",
                "source_key": "*.sav",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "hist"
            },
            {
                "source_node": "fit_rf",
                "source_key": "*.json",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "rf_pr_auc"
            },
            {
                "source_node": "fit_lgb",
                "source_key": "*.json",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "lgb_pr_auc"
            },
            {
                "source_node": "fit_cbc",
                "source_key": "*.json",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "cbc_pr_auc"
            },
            {
                "source_node": "fit_hist",
                "source_key": "*.json",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "hist_pr_auc"
            },
            {
                "source_node": "choose_best_model",
                "source_key": "*.sav",
                "source_type": "output",
                "target_node": "predict_explain",
                "target_type": "input",
                "target_key": "top_model"
            },
            {
                "source_node": "load_data_test",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "predict_explain",
                "target_type": "input",
                "target_key": "loaded_data"
            },
            {
                "source_node": "predict_explain",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "upload_to_s3",
                "target_type": "input",
                "target_key": "final_prediction"
            }
        ],
        "nodes": [
            {
                "name": "load_data_train",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "load_data_train",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.load_data(\"fit.sql\")'",
                    "inputs": {},
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {}
                }
            },
            {
                "name": "load_data_test",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "load_data_test",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.load_data(\"predict.sql\")'",
                    "inputs": {},
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {}
                }
            },
            {
                "name": "process_train",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "process_train",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.process_train()'",
                    "inputs": {
                        "loaded_data": []
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {}
                }
            },
            {
                "name": "fit_rf",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "fit_rf",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.fit_evaluate(\"rf\")'",
                    "inputs": {
                        "processed_data": []
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {}
                }
            },
            {
                "name": "fit_lgb",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "fit_lgb",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.fit_evaluate(\"lgb\")'",
                    "inputs": {
                        "processed_data": []
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {}
                }
            },
            {
                "name": "fit_cbc",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "fit_cbc",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.fit_evaluate(\"cbc\")'",
                    "inputs": {
                        "processed_data": []
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {}
                }
            },
            {
                "name": "fit_hist",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "fit_hist",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.fit_evaluate(\"hist\")'",
                    "inputs": {
                        "processed_data": []
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {}
                }
            },
            {
                "name": "choose_best_model",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "choose_best_model",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.choose_best_model()'",
                    "inputs": {
                        "rf": [],
                        "rf_pr_auc": [],
                        "lgb": [],
                        "lgb_pr_auc": [],
                        "cbc": [],
                        "cbc_pr_auc": [],
                        "hist": [],
                        "hist_pr_auc": []
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {}
                }
            },
            {
                "name": "predict_explain",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "predict_explain",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.predict_explain()'",
                    "inputs": {
                        "loaded_data": [],
                        "top_model": []
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {}
                }
            },
            {
                "name": "upload_to_s3",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "upload_to_s3",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.upload_to_s3()'",
                    "inputs": {
                        "final_prediction": []
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {}
                }
            }
        ],
        "project": "017b6d58-8fed-49a2-a934-58fdc93f1edd",
        "title": "pro_to_prox_first_pipeline"
    },
)
if resp.status_code == 400:
    raise RuntimeError(resp.json())
resp.raise_for_status()
data = resp.json()