import os
import torch
import pandas as pd
import numpy as np
from skimage import io

def get_file_paths(root_dir, break_output=False):
    ''' 
    Obtem o nome dos arquivos dentro uma sequencia de diretorio na forma:
    [ROOT_DIR]/[USER_ID]/[INTERVIEW_ID]/[AUDIO_CUT_FILENAME]
    
    break_output:
        [TRUE]:
        retorna uma lista com as tuplas ([ROOT_DIR], [USER_ID], [INTERVIEW_ID], [AUDIO_CUT_FILENAME])
        
        [FALSE]:
        retorna o caminho completo dos arquivos
    '''
    
    tuple_list, file_path = [], []
    
    for id in os.listdir(root_dir):
        id_path = os.path.join(root_dir, id)
        for rec in os.listdir(id_path):
            cut_path = os.path.join(id_path, rec)
            for cut in os.listdir(cut_path):
                tuple_list.append((root_dir, id, rec, cut))
                file_path.append(os.path.join(cut_path, cut))
    if break_output:
        return tuple_list
    else:
        return file_path

def get_user_dict(root_dir):
    '''
    Obtém o dicionário de formato {usuário: lista_arquivos} de `root_dir`
    '''
    tuple_list = get_file_paths(root_dir, break_output=True)
    user_list = np.unique([l[1] for l in tuple_list])
    user_dict = {u:[] for u in user_list}
    
    for item in tuple_list:
        user_dict[item[1]].append(os.path.join(*item))
    
    return user_dict

