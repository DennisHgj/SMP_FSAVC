import os.path
import random

import pandas as pd
from pandas import DataFrame



def FSL_sample_formatting(fewshot_csv, fewshot_test_csv, N_way, K_shot, caption='caption'):
    fewshot_df = pd.read_csv(fewshot_csv, header=None)
    fewshot_test_df = pd.read_csv(fewshot_test_csv, header=None)
    if caption == 'caption':
        fewshot_df.columns = ['name', 'cls', 'caption']
        fewshot_test_df.columns = ['name', 'cls', 'caption']
    elif caption == 'VGG_caption':
        fewshot_df.columns = ['name', 'cls', 'caption', 'label']
        fewshot_test_df.columns = ['name', 'cls', 'caption', 'label']
    elif caption == 'VGG_no_caption':
        fewshot_df.columns = ['name', 'cls', '_', 'label']
        fewshot_test_df.columns = ['name', 'cls', '_', 'label']
    else:
        fewshot_df.columns = ['name', 'cls', "_"]
        fewshot_test_df.columns = ['name', 'cls', "_"]

    if type(N_way) == int:
        class_col = fewshot_df['cls']
        class_set = set(class_col.tolist())
        class_list = list(class_set)
        random_select_class = random.sample(class_list, N_way)
        random_select_class.sort()
    elif type(N_way) == list:
        random_select_class = N_way

    print("selected class is {}".format(random_select_class))

    selected_class_test_df = fewshot_test_df[fewshot_test_df['cls'].isin(random_select_class)]

    selected_class_train_df = form_FS_training_df(random_select_class, fewshot_df, K_shot)

    return selected_class_train_df, selected_class_test_df, random_select_class


def form_FS_training_df(random_select_class, fewshot_df, K_shot):
    final_df = DataFrame()
    if K_shot == -1:
        for cls in random_select_class:
            temp_df = fewshot_df[fewshot_df['cls'] == cls]
            final_df = pd.concat([final_df, temp_df])
    else:
        for cls in random_select_class:
            temp_df = fewshot_df[fewshot_df['cls'] == cls]
            temp_df = temp_df.sample(K_shot)
            final_df = pd.concat([final_df, temp_df])
    return final_df