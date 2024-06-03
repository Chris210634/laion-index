import torch
import numpy as np
import os
import faiss
import time

image_features = torch.load('cache/20M_image_features_half.tensor').float()    # replace by absolute path to image features
caption_features = torch.load('cache/20M_caption_features_half.tensor').float()  # replace by absolute path to caption features

_, max_inds = torch.load('cache/closest_image_to_caption_tup.20M')

d = features.shape[1]

# Configure the IndexIVFFlat index
nlist = 131072  # number of clusters
# nlist = 10000
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
assert not index.is_trained

res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
faiss.omp_set_num_threads(8)

assert not gpu_index.is_trained

start_time = time.time()

gpu_index.train_paired(features[max_inds], caption_features) ###

end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

assert gpu_index.is_trained
cpu_index = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(cpu_index, 'cache/images.131072.ivf.paired_kmeans.index')

# QUANTIZE
_quantize(
    image_features,
    input_index_name='cache/images.131072.ivf.paired_kmeans.index', 
    output_index_name='cache/images.131072.QT_4bit.paired_kmeans.index', 
    quantization=faiss.ScalarQuantizer.QT_4bit
)





