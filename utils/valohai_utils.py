import json
import pickle


def load_score_and_model(model_name):
    model = pickle.load(open('/valohai/inputs/' + str(model_name) + '/' + str(model_name) + '.sav', 'rb'))
    f = open('/valohai/inputs/' + str(model_name) + '_pr_auc' + '/' + str(model_name) + '_pr_auc' + '.json', "r")
    model_pr_auc = json.loads(f.read())['pr_auc']
    return model, model_pr_auc
