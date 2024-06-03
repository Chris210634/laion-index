import numpy as np
import torch
import pandas as pd
import os, sys
import faiss
import torch.nn.functional as F
import numpy as np
import torch
import pandas as pd
import os
import faiss
from tqdm import tqdm
import time

def _mangle(i):
    s = str(i)
    if len(s) == 1:
        return '0' + s
    else:
        assert len(s) == 2
        return s
    
def _mangle4(n):
    s = str(n)
    return (4-len(s))*'0' + s

def _get_numpy_size_dic():
    d = {}
    with open('laion-2B-en.npy.filesizes.txt') as f:
        for line in f:
            ll = line.strip().split(',')
            assert not ll[0].strip() in d
            d[ll[0].strip()] = int(ll[1].strip())
    return d

def _get_nb_vectors_numpy(numpy_id, numpy_size_dic):
    size = numpy_size_dic['img_emb_{}.npy'.format(_mangle4(numpy_id))]
    assert (size - 128) % (2 * 768) == 0
    return (size - 128) // 2 // 768

def _get_numpy_start_position(i, numpy_size_dic):
    accum = 0
    for j in range(i):
        accum += _get_nb_vectors_numpy(j, numpy_size_dic)
    return accum

def _get_knn_start_position(i):
    accum = 0
    for j in range(i):
        accum += int(_get_json_info(j)['nb vectors'])
    return accum

def _get_knn_index_id(ind):
    start_index = 0
    knn_index_id = 0
    while start_index <= ind:
        start_index += int(_get_json_info(knn_index_id)['nb vectors'])
        knn_index_id += 1
    knn_index_id = knn_index_id-1
    return knn_index_id

def _get_reconstructed_vector(ind):
    start_index = 0
    knn_index_id = 0
    while start_index <= ind:
        start_index += int(_get_json_info(knn_index_id)['nb vectors'])
        knn_index_id += 1
    knn_index_id = knn_index_id-1
    print('knn_index_id: ', knn_index_id)
    
    index = _get_knn_index(knn_index_id)
    
    print('vecs left: ', index.ntotal - (ind - _get_knn_start_position(knn_index_id)))
    
    return torch.tensor(index.reconstruct_n(ind, 1))

def _get_reconstructed_vectors_multiple(ind, N):
    start_index = 0
    knn_index_id = 0
    while start_index <= ind:
        start_index += int(_get_json_info(knn_index_id)['nb vectors'])
        knn_index_id += 1
    knn_index_id = knn_index_id-1
    print('knn_index_id: ', knn_index_id)
    
    index = _get_knn_index(knn_index_id)
    
    print('vecs left: ', index.ntotal - (ind - _get_knn_start_position(knn_index_id)))
    assert N <  index.ntotal - (ind - _get_knn_start_position(knn_index_id))
    
    return torch.tensor(index.reconstruct_n(ind, N))

def _get_json_info(i):
    import json
    fn = '/projectnb/textconv/cliao25/LLAVA/clip-training/laion-2B-en/infos.json{}'.format(_mangle(i))
    with open(fn) as json_file:
        data = json.load(json_file)
    return data

def _get_knn_index(i):
    index = faiss.read_index('/projectnb/textconv/cliao25/LLAVA/clip-training/laion-2B-en/knn.index{}'.format(_mangle(i)))
    assert index.ntotal == int(_get_json_info(i)['nb vectors'])
    return index

def _get_root_index():
    root_index_name = 'cache/images.131072.QT_4bit.paired_kmeans.index'
    return faiss.read_index(root_index_name)
    
numpy_size_dic = _get_numpy_size_dic()

missingList = [
    667,
    904,
    1055,
    1357,
    1438,
    1552,
    1559,
    1615,
    1640,
    1668,
    1689,
    1798,
    1865,
    2044,
    2129,
    2261
]

for missing_num in missingList:
    st = _get_numpy_start_position(missing_num, numpy_size_dic)
    tt = _get_nb_vectors_numpy(missing_num, numpy_size_dic)
    f = F.normalize(_get_reconstructed_vectors_multiple(st, tt)).half().numpy()
    savename = 'cache/img_emb_{}.npy'.format(_mangle4(missing_num))
    np.save(savename, f)