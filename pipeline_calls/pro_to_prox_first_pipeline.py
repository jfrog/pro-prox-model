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
                "source_key": "rf.sav",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "rf"
            },
            {
                "source_node": "fit_lgb",
                "source_key": "lgb.sav",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "lgb"
            },
            {
                "source_node": "fit_cbc",
                "source_key": "cbc.sav",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "cbc"
            },
            {
                "source_node": "fit_hist",
                "source_key": "hist.sav",
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
                "source_node": "fit_rf",
                "source_key": "rf_columns.sav",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "rf_columns"
            },
            {
                "source_node": "fit_lgb",
                "source_key": "lgb_columns.sav",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "lgb_columns"
            },
            {
                "source_node": "fit_cbc",
                "source_key": "cbc_columns.sav",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "cbc_columns"
            },
            {
                "source_node": "fit_hist",
                "source_key": "hist_columns.sav",
                "source_type": "output",
                "target_node": "choose_best_model",
                "target_type": "input",
                "target_key": "hist_columns"
            },
            {
                "source_node": "choose_best_model",
                "source_key": "top_model.sav",
                "source_type": "output",
                "target_node": "predict_explain",
                "target_type": "input",
                "target_key": "top_model"
            },
            {
                "source_node": "choose_best_model",
                "source_key": "top_model_cols.sav",
                "source_type": "output",
                "target_node": "predict_explain",
                "target_type": "input",
                "target_key": "top_model_cols"
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
                    "environment": "0188286d-c924-c1c2-c42f-5668553cc8e2",
                    "commit": "prod",
                    "step": "load_data_train",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.load_data(\"fit.sql\")'",
                    "inputs": {},
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "environment_variable_groups": [],
                    "tags": ["prod"],
                    "time_limit": 0,
                    "environment_variables": {}
                },
                "on_error": "stop-all"
            },
            {
                "name": "load_data_test",
                "type": "execution",
                "template": {
                    "environment": "0188286d-c924-c1c2-c42f-5668553cc8e2",
                    "commit": "prod",
                    "step": "load_data_test",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.load_data(\"predict.sql\")'",
                    "inputs": {},
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "environment_variable_groups": [],
                    "tags": ["prod"],
                    "time_limit": 0,
                    "environment_variables": {}
                },
                "on_error": "stop-all"
            },
            {
                "name": "process_train",
                "type": "execution",
                "template": {
                    "environment": "0188286d-c924-c1c2-c42f-5668553cc8e2",
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
                    "environment_variable_groups": [],
                    "tags": ["prod"],
                    "time_limit": 0,
                    "environment_variables": {}
                },
                "on_error": "stop-all"
            },
            {
                "name": "fit_rf",
                "type": "execution",
                "template": {
                    "environment": "0188286d-c924-c1c2-c42f-5668553cc8e2",
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
                    "environment_variable_groups": [],
                    "tags": ["prod"],
                    "time_limit": 0,
                    "environment_variables": {}
                },
                "on_error": "stop-all"
            },
            {
                "name": "fit_lgb",
                "type": "execution",
                "template": {
                    "environment": "0188286d-c924-c1c2-c42f-5668553cc8e2",
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
                    "environment_variable_groups": [],
                    "tags": ["prod"],
                    "time_limit": 0,
                    "environment_variables": {}
                },
                "on_error": "stop-all"
            },
            {
                "name": "fit_cbc",
                "type": "execution",
                "template": {
                    "environment": "0188286d-c924-c1c2-c42f-5668553cc8e2",
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
                    "environment_variable_groups": [],
                    "tags": ["prod"],
                    "time_limit": 0,
                    "environment_variables": {}
                },
                "on_error": "stop-all"
            },
            {
                "name": "fit_hist",
                "type": "execution",
                "template": {
                    "environment": "0188286d-c924-c1c2-c42f-5668553cc8e2",
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
                    "environment_variable_groups": [],
                    "tags": ["prod"],
                    "time_limit": 0,
                    "environment_variables": {}
                },
                "on_error": "stop-all"
            },
            {
                "name": "choose_best_model",
                "type": "execution",
                "template": {
                    "environment": "0188286d-c924-c1c2-c42f-5668553cc8e2",
                    "commit": "prod",
                    "step": "choose_best_model",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.choose_best_model()'",
                    "inputs": {
                        "rf": [],
                        "lgb": [],
                        "cbc": [],
                        "hist": [],
                        "rf_pr_auc": [],
                        "lgb_pr_auc": [],
                        "cbc_pr_auc": [],
                        "hist_pr_auc": [],
                        "rf_columns": [],
                        "lgb_columns": [],
                        "cbc_columns": [],
                        "hist_columns": []
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "environment_variable_groups": [],
                    "tags": ["prod"],
                    "time_limit": 0,
                    "environment_variables": {}
                },
                "on_error": "stop-all"
            },
            {
                "name": "predict_explain",
                "type": "execution",
                "template": {
                    "environment": "0188286d-c924-c1c2-c42f-5668553cc8e2",
                    "commit": "prod",
                    "step": "predict_explain",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import prod_valohai; prod_valohai.predict_explain()'",
                    "inputs": {
                        "loaded_data": [],
                        "top_model": [],
                        "top_model_cols": []
                    },
                    "parameters": {},
                    "runtime_config": {},
                    "inherit_environment_variables": True,
                    "environment_variable_groups": [],
                    "tags": ["prod"],
                    "time_limit": 0,
                    "environment_variables": {}
                },
                "on_error": "stop-all"
            },
            {
                "name": "upload_to_s3",
                "type": "execution",
                "template": {
                    "environment": "0188286d-c924-c1c2-c42f-5668553cc8e2",
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
                    "environment_variable_groups": [],
                    "tags": ["prod"],
                    "time_limit": 0,
                    "environment_variables": {}
                },
                "on_error": "stop-all"
            }
        ],
        "project": "017b6d58-8fed-49a2-a934-58fdc93f1edd",
        "tags": [
            "prod"
        ],
        "parameters": {},
        "title": "pro_to_prox_first_pipeline"
    },
)
if resp.status_code == 400:
    raise RuntimeError(resp.json())
resp.raise_for_status()
data = resp.json()