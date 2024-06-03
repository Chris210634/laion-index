import torch
image_features = torch.load('cache/20M_image_features_half.tensor')      # replace by absolute path to image features
caption_features = torch.load('cache/20M_caption_features_half.tensor')  # replace by absolute path to caption features

import time

def _bmm_max(A, B, bs=100):
    n = len(A)
    ptr = 0
    t = torch.zeros((n, )).long()
    r = torch.zeros((n, )).float()
    B_cuda = B.cuda().half()
    while ptr < n:
        begin = ptr
        end = min(n, ptr+bs)
        max_result = (A[begin:end].cuda().half() @ B_cuda.T).max(1)
        t[begin:end] = max_result.indices.cpu()
        r[begin:end] = max_result.values.cpu()
        ptr = end
    
    assert ptr == n
    return r, t

N = image_features.shape[0]
ptr = 0
bs = 1000000

closest_i_to_c_tup_list = []
while ptr < N:
    print('progress: {}M / {}M'.format(ptr//1000000, N//1000000))
    start_time = time.time()
    
    begin = ptr
    end = min(N, begin+bs)
    closest_i_to_c = _bmm_max(caption_features , image_features[begin:end])
    closest_i_to_c_tup_list.append((closest_i_to_c[0], closest_i_to_c[1] + begin))
    ptr = end

    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    
  
# values, indices
torch.set_num_threads(8)
max_inds = torch.stack([closest_i_to_c_tup_list[i][0] for i in range(len(closest_i_to_c_tup_list))]).T.max(1).indices
max_vals = torch.stack([closest_i_to_c_tup_list[i][0] for i in range(len(closest_i_to_c_tup_list))]).T.max(1).values
master_inds = torch.stack([closest_i_to_c_tup_list[i][1] for i in range(len(closest_i_to_c_tup_list))]).T
max_inds = torch.gather(master_inds, 1, max_inds.unsqueeze(0).T).view(-1)
torch.save((max_vals, max_inds), 'cache/closest_image_to_caption_tup.20M')