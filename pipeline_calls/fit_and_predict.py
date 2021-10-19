import requests

resp = requests.request(
    url="https://valohai-prod-is.jfrog.org/api/v0/pipelines/",
    method="POST",
    headers={"Authorization": "Token YOUR_TOKEN_HERE"},
    json={
        "edges": [
            {
                "source_node": "load_data_fit",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "consolidate_opps",
                "target_type": "input",
                "target_key": "loaded_data"
            },
            {
                "source_node": "consolidate_opps",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "process_df_fit",
                "target_type": "input",
                "target_key": "loaded_data"
            },
            {
                "source_node": "process_df_fit",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "fit_rf",
                "target_type": "input",
                "target_key": "processed_data"
            },
            {
                "source_node": "process_df_fit",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "fit_etc",
                "target_type": "input",
                "target_key": "processed_data"
            },
            {
                "source_node": "process_df_fit",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "fit_cbc",
                "target_type": "input",
                "target_key": "processed_data"
            },
            {
                "source_node": "process_df_fit",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "fit_hist",
                "target_type": "input",
                "target_key": "processed_data"
            },
            {
                "source_node": "process_df_fit",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "make_bars",
                "target_type": "input",
                "target_key": "processed_data"
            },
            {
                "source_node": "fit_rf",
                "source_key": "*.sav",
                "source_type": "output",
                "target_node": "choose",
                "target_type": "input",
                "target_key": "rf"
            },
            {
                "source_node": "fit_etc",
                "source_key": "*.sav",
                "source_type": "output",
                "target_node": "choose",
                "target_type": "input",
                "target_key": "etc"
            },
            {
                "source_node": "fit_cbc",
                "source_key": "*.sav",
                "source_type": "output",
                "target_node": "choose",
                "target_type": "input",
                "target_key": "cbc"
            },
            {
                "source_node": "fit_hist",
                "source_key": "*.sav",
                "source_type": "output",
                "target_node": "choose",
                "target_type": "input",
                "target_key": "hist"
            },
            {
                "source_node": "fit_rf",
                "source_key": "*.json",
                "source_type": "output",
                "target_node": "choose",
                "target_type": "input",
                "target_key": "rf_pr_auc"
            },
            {
                "source_node": "fit_etc",
                "source_key": "*.json",
                "source_type": "output",
                "target_node": "choose",
                "target_type": "input",
                "target_key": "etc_pr_auc"
            },
            {
                "source_node": "fit_cbc",
                "source_key": "*.json",
                "source_type": "output",
                "target_node": "choose",
                "target_type": "input",
                "target_key": "cbc_pr_auc"
            },
            {
                "source_node": "fit_hist",
                "source_key": "*.json",
                "source_type": "output",
                "target_node": "choose",
                "target_type": "input",
                "target_key": "hist_pr_auc"
            },
            {
                "source_node": "choose",
                "source_key": "*.sav",
                "source_type": "output",
                "target_node": "predict",
                "target_type": "input",
                "target_key": "top_model"
            },
            {
                "source_node": "make_bars",
                "source_key": "low_bar_for_predict.csv",
                "source_type": "output",
                "target_node": "predict",
                "target_type": "input",
                "target_key": "low_bar_for_predict"
            },
            {
                "source_node": "make_bars",
                "source_key": "high_bar_for_predict.csv",
                "source_type": "output",
                "target_node": "predict",
                "target_type": "input",
                "target_key": "high_bar_for_predict"
            },
            {
                "source_node": "load_data_predict",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "process_df_predict",
                "target_type": "input",
                "target_key": "loaded_data"
            },
            {
                "source_node": "process_df_predict",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "predict",
                "target_type": "input",
                "target_key": "processed_data"
            },
            {
                "source_node": "predict",
                "source_key": "*.csv",
                "source_type": "output",
                "target_node": "upload_to_s3",
                "target_type": "input",
                "target_key": "final_prediction"
            }
        ],
        "nodes": [
            {
                "name": "load_data_fit",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "load_data_fit",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import main; main.load_data(\"fit.sql\")'",
                    "inputs": {},
                    "parameters": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {},
                    "runtime_config": {
                        "type": "kubernetes",
                        "containers": {
                            "workload": {
                                "resources": {
                                    "requests": {
                                        "cpu": "1.0",
                                        "memory": "32"
                                    },
                                    "limits": {
                                        "cpu": "",
                                        "memory": ""
                                    }
                                }
                            }
                        }
                    }
                },
                "actions": []
            },
            {
                "name": "load_data_predict",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "load_data_predict",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import main; main.load_data(\"predict.sql\")'",
                    "inputs": {},
                    "parameters": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {},
                    "runtime_config": {
                        "type": "kubernetes",
                        "containers": {
                            "workload": {
                                "resources": {
                                    "requests": {
                                        "cpu": "1.0",
                                        "memory": "32"
                                    },
                                    "limits": {
                                        "cpu": "",
                                        "memory": ""
                                    }
                                }
                            }
                        }
                    }
                },
                "actions": []
            },
            {
                "name": "consolidate_opps",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "consolidate_opps",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import main; main.consolidate_opps()'",
                    "inputs": {
                        "loaded_data": []
                    },
                    "parameters": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {},
                    "runtime_config": {
                        "type": "kubernetes",
                        "containers": {
                            "workload": {
                                "resources": {
                                    "requests": {
                                        "cpu": "1.0",
                                        "memory": "32"
                                    },
                                    "limits": {
                                        "cpu": "",
                                        "memory": ""
                                    }
                                }
                            }
                        }
                    }
                },
                "actions": []
            },
            {
                "name": "process_df_fit",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "process_df",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import main; main.process_df()'",
                    "inputs": {
                        "loaded_data": []
                    },
                    "parameters": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {},
                    "runtime_config": {
                        "type": "kubernetes",
                        "containers": {
                            "workload": {
                                "resources": {
                                    "requests": {
                                        "cpu": "1.0",
                                        "memory": "32"
                                    },
                                    "limits": {
                                        "cpu": "",
                                        "memory": ""
                                    }
                                }
                            }
                        }
                    }
                },
                "actions": []
            },
            {
                "name": "process_df_predict",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "process_df",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import main; main.process_df()'",
                    "inputs": {
                        "loaded_data": []
                    },
                    "parameters": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {},
                    "runtime_config": {
                        "type": "kubernetes",
                        "containers": {
                            "workload": {
                                "resources": {
                                    "requests": {
                                        "cpu": "1.0",
                                        "memory": "32"
                                    },
                                    "limits": {
                                        "cpu": "",
                                        "memory": ""
                                    }
                                }
                            }
                        }
                    }
                },
                "actions": []
            },
            {
                "name": "fit_rf",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "fit_rf",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import main; main.fit(\"rf\")'",
                    "inputs": {
                        "processed_data": []
                    },
                    "parameters": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {},
                    "runtime_config": {
                        "type": "kubernetes",
                        "containers": {
                            "workload": {
                                "resources": {
                                    "requests": {
                                        "cpu": "1.0",
                                        "memory": "32"
                                    },
                                    "limits": {
                                        "cpu": "",
                                        "memory": ""
                                    }
                                }
                            }
                        }
                    }
                },
                "actions": []
            },
            {
                "name": "fit_etc",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "fit_etc",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import main; main.fit(\"etc\")'",
                    "inputs": {
                        "processed_data": []
                    },
                    "parameters": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {},
                    "runtime_config": {
                        "type": "kubernetes",
                        "containers": {
                            "workload": {
                                "resources": {
                                    "requests": {
                                        "cpu": "1.0",
                                        "memory": "32"
                                    },
                                    "limits": {
                                        "cpu": "",
                                        "memory": ""
                                    }
                                }
                            }
                        }
                    }
                },
                "actions": []
            },
            {
                "name": "fit_cbc",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "fit_cbc",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import main; main.fit(\"cbc\")'",
                    "inputs": {
                        "processed_data": []
                    },
                    "parameters": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {},
                    "runtime_config": {
                        "type": "kubernetes",
                        "containers": {
                            "workload": {
                                "resources": {
                                    "requests": {
                                        "cpu": "1.0",
                                        "memory": "32"
                                    },
                                    "limits": {
                                        "cpu": "",
                                        "memory": ""
                                    }
                                }
                            }
                        }
                    }
                },
                "actions": []
            },
            {
                "name": "fit_hist",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "fit_hist",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import main; main.fit(\"hist\")'",
                    "inputs": {
                        "processed_data": []
                    },
                    "parameters": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {},
                    "runtime_config": {
                        "type": "kubernetes",
                        "containers": {
                            "workload": {
                                "resources": {
                                    "requests": {
                                        "cpu": "1.0",
                                        "memory": "32"
                                    },
                                    "limits": {
                                        "cpu": "",
                                        "memory": ""
                                    }
                                }
                            }
                        }
                    }
                },
                "actions": []
            },
            {
                "name": "choose",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "choose",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import main; main.choose()'",
                    "inputs": {
                        "rf": [],
                        "rf_pr_auc": [],
                        "etc": [],
                        "etc_pr_auc": [],
                        "cbc": [],
                        "cbc_pr_auc": [],
                        "hist": [],
                        "hist_pr_auc": []
                    },
                    "parameters": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {},
                    "runtime_config": {
                        "type": "kubernetes",
                        "containers": {
                            "workload": {
                                "resources": {
                                    "requests": {
                                        "cpu": "1.0",
                                        "memory": "32"
                                    },
                                    "limits": {
                                        "cpu": "",
                                        "memory": ""
                                    }
                                }
                            }
                        }
                    }
                },
                "actions": []
            },
            {
                "name": "make_bars",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "make_bars",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import main; main.make_bars()'",
                    "inputs": {
                        "processed_data": []
                    },
                    "parameters": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {},
                    "runtime_config": {
                        "type": "kubernetes",
                        "containers": {
                            "workload": {
                                "resources": {
                                    "requests": {
                                        "cpu": "1.0",
                                        "memory": "32"
                                    },
                                    "limits": {
                                        "cpu": "",
                                        "memory": ""
                                    }
                                }
                            }
                        }
                    }
                },
                "actions": []
            },
            {
                "name": "predict",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "predict",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import main; main.predict()'",
                    "inputs": {
                        "processed_data": [],
                        "low_bar_for_predict": [],
                        "high_bar_for_predict": [],
                        "top_model": []
                    },
                    "parameters": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {},
                    "runtime_config": {
                        "type": "kubernetes",
                        "containers": {
                            "workload": {
                                "resources": {
                                    "requests": {
                                        "cpu": "1.0",
                                        "memory": "32"
                                    },
                                    "limits": {
                                        "cpu": "",
                                        "memory": ""
                                    }
                                }
                            }
                        }
                    }
                },
                "actions": []
            },
            {
                "name": "upload_to_s3",
                "type": "execution",
                "template": {
                    "environment": "01742a18-07ca-75b6-1a1f-f8cc93b058a0",
                    "commit": "prod",
                    "step": "upload_to_s3",
                    "image": "yotamljfrog/proprox:0.1",
                    "command": "pip install -r requirements.txt\npython -c 'import main; main.upload_to_s3()'",
                    "inputs": {
                        "final_prediction": []
                    },
                    "parameters": {},
                    "inherit_environment_variables": True,
                    "time_limit": 0,
                    "environment_variables": {},
                    "runtime_config": {
                        "type": "kubernetes",
                        "containers": {
                            "workload": {
                                "resources": {
                                    "requests": {
                                        "cpu": "1.0",
                                        "memory": "32"
                                    },
                                    "limits": {
                                        "cpu": "",
                                        "memory": ""
                                    }
                                }
                            }
                        }
                    }
                },
                "actions": []
            }
        ],
        "project": "017b6d58-8fed-49a2-a934-58fdc93f1edd",
        "title": "fit_and_predict"
    },
)
if resp.status_code == 400:
    raise RuntimeError(resp.json())
resp.raise_for_status()
data = resp.json()