import os
import configparser
import itertools
import datetime
import time

EvalMethodFofDBoW3 = ['brisk','orb']
EvalMethodFofDBoW2 = ['SURF','proposed method']
ResultDir = '/home/dongwonshin/Desktop/LoopClosureDetectionTestBed/result_folder'
DepthLevel = 5

def configWritingForDBoW2(eval_method,eval_dataset, cluster_center):
    config['Experiment_parameters'] = {}
    config['Experiment_parameters']['eval_method'] = eval_method
    config['Experiment_parameters']['eval_dataset'] = eval_dataset
    config['Experiment_parameters']['eval_desc'] = 'survey_final_512dim' #'all_desc'

    # 'bigger_feature_size': 1024 , large-scale-training:512
    config['Experiment_parameters']['network_model'] = 'large-scale-training'
    config['Experiment_parameters']['scoring_type'] = 'L1_NORM'
    config['Experiment_parameters']['cluster_center'] = str(cluster_center)
    config['Experiment_parameters']['depth_level'] = str(DepthLevel)
    config['Experiment_parameters']['result_dir'] = ResultDir

    conf_file = 'input_config.ini'
    with open(conf_file, 'w') as configfile:
        config.write(configfile)

def configWritingForDBoW3(eval_method,eval_dataset, cluster_center):
    config['Experiment_parameters'] = {}
    config['Experiment_parameters']['eval_method'] = eval_method
    config['Experiment_parameters']['eval_dataset'] = eval_dataset
    config['Experiment_parameters']['scoring_type'] = 'L1_NORM'
    config['Experiment_parameters']['cluster_center'] = str(cluster_center)
    config['Experiment_parameters']['depth_level'] = str(DepthLevel)
    config['Experiment_parameters']['result_dir'] = ResultDir

    conf_file = 'input_config.ini'
    with open(conf_file, 'w') as configfile:
        config.write(configfile)


def DBoW3Experiment(eval_method, eval_dataset, cluster_center):
    configWritingForDBoW3(eval_method, eval_dataset, cluster_center)
    os.system("/home/dongwonshin/Desktop/DBow3/build2/utils/demo_general")

def DBoW2Experiment(eval_method, eval_dataset, cluster_center):
    configWritingForDBoW2(eval_method, eval_dataset, cluster_center)
    os.system("/home/dongwonshin/Desktop/DBoW2/build2/demo")


def StartHere():
    now = datetime.datetime.now()
    date_string = now.strftime('%Y-%m-%d-%H-%M-%S')
    # print(ResultDir + '/%s[Start Here]' % date_string)
    os.mkdir(ResultDir + '/%s[Start Here]' % date_string)

    time.sleep(1)

def EndHere():
    now = datetime.datetime.now()
    date_string = now.strftime('%Y-%m-%d-%H-%M-%S')
    # print(ResultDir + '/%s[Start Here]' % date_string)
    os.mkdir(ResultDir + '/%s[End Here]' % date_string)


if __name__ == '__main__':

    StartHere()

    config = configparser.ConfigParser()

    eval_methods = ['proposed method']
    cluster_centers = [10,5,15]
    eval_datasets = ['City Centre','New College']
    # eval_datasets = ['KAIST_All_Day(West)', 'KAIST_All_Day(East)','KAIST_All_Day(North)']

    for eval_method, eval_dataset, cluster_center in itertools.product(eval_methods, eval_datasets,cluster_centers):

        if (eval_method in EvalMethodFofDBoW2):
            DBoW2Experiment(eval_method, eval_dataset, cluster_center)

        if (eval_method in EvalMethodFofDBoW3):
            DBoW3Experiment(eval_method, eval_dataset, cluster_center)

    EndHere()