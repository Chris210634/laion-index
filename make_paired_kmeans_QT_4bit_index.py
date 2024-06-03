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
import subprocess

numpy_cache_dir = 'cache'   # change to a directory that can temporarily hold 100 GB of data
output_dir = 'cache'        # change to a directory that can hold ~ 1 TB of data

# # make a list of numpys to size
# b = False
# cum_size = 0
# with open('laion-2B-en.npy.filesizes.txt', 'w') as g:
#     with open('spider.out') as f:
#         for line in f:
#             if line.strip().endswith('.npy'):
#                 assert not b
#                 numpy = line.strip().split('/')[-1]
#                 b = True
#             elif line.strip().endswith('[application/octet-stream]'):
#                 assert b
#                 size = int(line.strip().split(':')[1].strip().split('(')[0].strip())
#                 assert (size - 128) % (2 * 768) == 0
#                 cum_size += (size - 128) // 2 // 768
#                 b = False
#                 g.write('{}, {}\n'.format(numpy, size))
#             elif 'broken link!!!' in line:
#                 print(numpy)
#                 assert b
#                 size = 0
#                 b = False
#                 g.write('{}, {}\n'.format(numpy, size))

# def _get_reconstructed_vector(ind):
#     start_index = 0
#     knn_index_id = 0
#     while start_index <= ind:
#         start_index += int(_get_json_info(knn_index_id)['nb vectors'])
#         knn_index_id += 1
#     knn_index_id = knn_index_id-1
    
#     index = _get_knn_index(knn_index_id)
#     return torch.tensor(index.reconstruct_n(ind, 1))

# def _get_knn_index(i):
#     index = faiss.read_index('/projectnb/textconv/cliao25/LLAVA/clip-training/laion-2B-en/knn.index{}'.format(_mangle(i)))
#     assert index.ntotal == int(_get_json_info(i)['nb vectors'])
#     return index

# def _get_json_info(i):
#     import json
#     fn = '/projectnb/textconv/cliao25/LLAVA/clip-training/laion-2B-en/infos.json{}'.format(_mangle(i))
#     with open(fn) as json_file:
#         data = json.load(json_file)
#     return data

# def _get_knn_start_position(i):
#     accum = 0
#     for j in range(i):
#         accum += int(_get_json_info(j)['nb vectors'])
#     return accum

# def _get_numpy_vector(ind, numpy_size_dic):
#     start_index = 0
#     numpy_id = 0
#     while start_index <= ind:
#         start_index += _get_nb_vectors_numpy(numpy_id, numpy_size_dic)
#         numpy_id += 1
#     numpy_id = numpy_id-1
# #     return numpy_id
#     start = _get_numpy_start_position(numpy_id, numpy_size_dic)
#     filepath = 'cache/img_emb_{}.npy'.format(_mangle4(numpy_id))
#     print(filepath)
#     return torch.tensor(np.load(filepath)[ind-start]).unsqueeze(0)

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

def _get_root_index():
    root_index_name = 'cache/images.131072.QT_4bit.paired_kmeans.index'
    return faiss.read_index(root_index_name)

##################################################################
##################################################################
##################################################################

numpy_size_dic = _get_numpy_size_dic()

res = faiss.StandardGpuResources()
co = faiss.GpuClonerOptions()
co.useFloat16 = True
root_index = _get_root_index()
root_index = faiss.index_cpu_to_gpu(res, 0, root_index, co)
faiss.omp_set_num_threads(8)
torch.set_num_threads(8)

INDEX = int(sys.argv[1])

from_numpy_id = INDEX * 40
to_numpy_id = min(from_numpy_id + 40, 2314)

print('RUNNNING INDEX #{} from numpy {} to {}'.format(INDEX, from_numpy_id, to_numpy_id))

START = _get_numpy_start_position(from_numpy_id, numpy_size_dic)

for numpy_id in range(from_numpy_id, to_numpy_id):
    print('progress: {}/{} '.format(numpy_id-from_numpy_id, 40))
    
    numpy_name = 'img_emb_{}.npy'.format(_mangle4(numpy_id))
    filepath = os.path.join(numpy_cache_dir, numpy_name)
    
    # download from LAION server if not present
    start_time = time.time()
    if not os.path.exists(filepath):
        subprocess.run(
            'wget -P {} \
            https://the-eye.eu/public/AI/cah/laion5b/embeddings/laion2B-en/img_emb/{}'.format(numpy_cache_dir, numpy_name), 
            capture_output=False, text=True, shell=True
        )
    assert os.path.exists(filepath)
    end_time = time.time()
    execution_time = end_time - start_time
    print("download time:", execution_time, "seconds")
    
    assert os.path.getsize(filepath) == numpy_size_dic[numpy_name]
    print(filepath)
    f = np.load(filepath)

    start_time = time.time()
    root_index.add(f)
    end_time = time.time()
    execution_time = end_time - start_time
    print("add time:", execution_time, "seconds")
    
    os.remove(filepath)
    
    del f

# add offset
output_index_name = os.path.join(output_dir, 'knn.paired_QT_4bit.index{}'.format(INDEX))
cpu_index = faiss.index_gpu_to_cpu(root_index)
zero_index = _get_root_index()
zero_index.merge_from(cpu_index, START)
zero_index.nprobe = 8
faiss.write_index(zero_index, output_index_name)