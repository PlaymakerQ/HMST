import json
import os
import datetime
import logging
import yaml
import numpy as np

valid_metrics = ['Acc', 'MAP', 'NDCG', 'MRR']

def save_args_to_json(args, set_file):
    # set parameters to dict.
    args_dict = vars(args)
    set_file = os.path.join(set_file, 'settings.json')
    # save json
    with open(set_file, 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)
    return set_file


def metric_string(metric_dict):
    metric_str = ""
    score_str = ""

    for metric in metric_dict:

        if metric == 'MRR':
            score = metric_dict[metric]
            score = f"{score:.4f}"
            fix_len = len(score)
            metric_str += metric.ljust(fix_len) + " | "
            score_str += score.rjust(fix_len) + " | "
        else:
            for k in metric_dict[metric].keys():
                score = metric_dict[metric][k]
                score = f"{score:.4f}"
                metric_k = f"{metric}@{k:02d}"
                fix_len = len(metric_k)
                metric_str += metric_k.ljust(fix_len) + " | "
                score_str += score.rjust(fix_len) + " | "

    return metric_str, score_str


class DictAsObject:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)


class MyTool:

    @staticmethod
    def get_root_path():
        return os.path.dirname(os.path.dirname(__file__))

    @staticmethod
    def set_save_path(data_name, model_name, save_root_name='save'):
        root_path = MyTool.get_root_path()
        save_root_path = os.path.join(root_path, save_root_name)
        if not os.path.exists(save_root_path):
            os.mkdir(save_root_path)
        save_path = os.path.join(save_root_path, data_name)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        dt = datetime.datetime.now()
        folder_name = dt.strftime("%m%d_%H%M%S_") + model_name
        folder_path = os.path.join(save_path, folder_name)
        os.mkdir(folder_path)
        return folder_path

    @staticmethod
    def save_config(config, save_path):
        pass

    @staticmethod
    def set_logging(save_path, model_name=None):
        """ set log file path and logging formats"""
        log_format = f"%(asctime)s | {model_name} | %(levelname)-4s | %(message)s"
        formatter = logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logging.getLogger("").addHandler(console)
        logging.getLogger("").setLevel(logging.INFO)

        # save in log file
        filename = os.path.join(save_path, 'run.log')
        file = logging.FileHandler(filename)
        file.setFormatter(formatter)
        logging.getLogger("").addHandler(file)

    @staticmethod
    def convert_dict_to_object(dict):
        return DictAsObject(dict)

    @staticmethod
    def load_model_params(data_name):
        config_path = os.path.join(MyTool.get_root_path(), "configs", f"{data_name}.yaml")
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

    @staticmethod
    def save_model_params(model_params, save_path):
        config_path = os.path.join(save_path, "config.yaml")
        with open(config_path, 'w') as file:
            yaml.dump(model_params, file)

    @staticmethod
    def get_metric_string(metric_dict):
        return metric_string(metric_dict)

    @staticmethod
    def print_metrics(metric_dict):
        metric_result = metric_string(metric_dict)
        logging.info(metric_result[0])
        logging.info(metric_result[1])

    @staticmethod
    def init_metric_dict(metric_list, k_list):
        metrics = {}
        for metric in metric_list:

            if metric in valid_metrics:

                if metric == 'MRR':
                    metrics[metric] = []
                else:
                    metrics[metric] = {}
                    for k in k_list:
                        metrics[metric][k] = []
        return metrics

    @staticmethod
    def get_average_metric(metric_dict):
        for metric in metric_dict.keys():

            if metric == 'MRR':
                metric_dict[metric] = np.mean(metric_dict[metric])
            else:
                for k in metric_dict[metric].keys():
                    metric_dict[metric][k] = np.mean(metric_dict[metric][k])
        return metric_dict



if __name__ == '__main__':

    pass