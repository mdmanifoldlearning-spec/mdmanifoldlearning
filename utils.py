import pickle 
import os 
import pandas as pd

def check_file_exist(path):
    """
    check if ``file`` exists
    :param path:
    :return: T/F
    """
    return os.path.isfile(path)
    
def dump_pickle(path, data):
    """
    save data as binary file
    :param path:
    :param data:
    :return:
    """
    dir_name = os.path.dirname(path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=4)  


def load_pickle(path):
    """
    :param path:
    :return:
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data    

def save_to_df(dict_to_save, df_path, replace_if_exist = False):
    """
    save a row of data to df.

    Parameters
    ----------
    dict_to_save : dict
        a dict that contains a row of df data. e.g. {'auc': 0.99, 'dim': 64}
    df_path : str
        df path to save.
    replace_if_exist : True/False, optional
        if replace the existing df.

    """
    dir_name = os.path.dirname(df_path)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if not replace_if_exist and os.path.exists(df_path):
        exist_df = pd.read_csv(df_path)
        df_to_save = pd.DataFrame(dict_to_save, index=[0])
        new_df = pd.concat([exist_df, df_to_save], ignore_index=True)
        new_df.to_csv(df_path, index = False)
    else:
        new_df = pd.DataFrame(dict_to_save, index=[0])
        new_df.to_csv(df_path, index = False)

if __name__ == '__main__':
    """
    test save_to_df
    """
    df_path = '../results/test.csv'
    dict_to_save = {'a':1, 'b':2.0, 'c':'hello'}
    dict_to_save2 = {'a':1, 'c':'how are you', 'd':-999}
    save_to_df(dict_to_save, df_path, replace_if_exist = False)
    save_to_df(dict_to_save2, df_path, replace_if_exist = True)



