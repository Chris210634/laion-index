# Implementation of Paried k-means Indexing
----------

## Step 1 Install Faiss

We use a slightly modified version of FAISS to implement paired k-means indexing. The fork [here](https://github.com/Chris210634/faiss) contains the modifications. It is forked from version 1.7.3 of Faiss.

```bash
git clone https://github.com/Chris210634/faiss
cd faiss
cmake -B build .
make -C build -j faiss
make -C build -j swigfaiss
cd build/faiss/python && python setup.py install --user
```

This faiss build was tested on linux with the following versions of packages:

```1) cmake/3.22.2   2) gcc/12.2.0   3) python3/3.8.10   4) cuda/12.2   5) pytorch/1.13.1   6) swig/4.0.2```

## Step 2 Calculate or Download Features

Now we need to download or calculate the features 

TODO

## Step 3 Train the index

Replace the following lines in `brute_force_nearest_neighbor.py` and `train_paired_kmeans_faiss_index.py` with the path to the image and caption features caluclated/downloaded from the previous step:

```python
image_features = torch.load('')    # replace by absolute path to image features
caption_features = torch.load('')  # replace by absolute path to caption features
```

First, calculate the nearest image feature to each caption feature (approx. 2-4 hours on A40):

```
mkdir cache
python brute_force_nearest_neighbor.py
```

The above code stores the cosine similarity and index of the closest caption feature to each image feature in `cache/closest_image_to_caption_tup.20M`.

Second, train the index using paired kmeans (about 1 hour on A40):

```
python train_paired_kmeans_faiss_index.py
```

The above code stores the empty trained index in `cache/images.131072.ivf.paired_kmeans.index`.


