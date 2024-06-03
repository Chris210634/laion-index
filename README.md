# Implementation of Paried k-means Indexing
----------

ArXiv paper: [link](https://arxiv.org/pdf/2402.04416)

![](https://github.com/Chris210634/laion-index/blob/main/tools/paired_kmeans_algo.PNG?raw=true)

## Step 1: Install Faiss

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

## Step 2: Calculate or Download Features

For clustering (index training), we used a 20M subset of LAION-2B. The indexing model is CLIP ViT/L-14.

You can download these features here: [20M_image_features_half.tensor](https://drive.google.com/file/d/1GC4K0_MegJg8wNE9rcnh5O149iMu4Rhp/view?usp=drive_link), [20M_caption_features_half.tensor](https://drive.google.com/file/d/1dGgzeqseleYR42Vd2UjLSPVTkZ_TSpii/view?usp=drive_link)

Alternatively, you can use the python scripts in `tools/` folder to generate the features from scratch. For example:

```bash
python tools/generate_text_features.py  --img_list_filename caption_list_filename
python tools/generate_image_features.py --img_list_filename image_list_filename
```
where `caption_list_filename` is a file created using `torch.save(...)` with a list of captions as strings. `image_list_filename` is a list of image file paths (probably safest to use absolute paths).

## Step 3: Train the index

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

The above code stores the empty trained index in `cache/images.131072.QT_4bit.paired_kmeans.index`.

## Step 4: Add features to the index

Now we need to add the LAION image features to the index. This step requires 1-2 days with a GPU depending on the download speed.

You may want to edit the following lines in `make_paired_kmeans_QT_4bit_index.py` to change the output directory for the indices, as the index files are large:

```python
numpy_cache_dir = 'cache'   # change to a directory that can temporarily hold 100 GB of data
output_dir = 'cache'        # change to a directory that can hold ~ 1 TB of data
```

Run:
```bash
for i in {0..57}
do
  python make_paired_kmeans_QT_4bit_index.py $i
done
```

This will populate the `cache` directory with the new faiss indices for LAION named `knn.paired_QT_4bit.index*`. 

### Note:

The following numpy files are missing from the LAION server:
```
img_emb_0667.npy
img_emb_0904.npy
img_emb_1055.npy
img_emb_1357.npy
img_emb_1438.npy
img_emb_1552.npy
img_emb_1559.npy
img_emb_1615.npy
img_emb_1640.npy
img_emb_1668.npy
img_emb_1689.npy
img_emb_1798.npy
img_emb_1865.npy
img_emb_2044.npy
img_emb_2129.npy
img_emb_2261.npy
```

Please download these from [this Google drive](https://drive.google.com/drive/folders/1LPn_UvkHPqEhUUdTYifnJflKPVar1uNm?usp=drive_link)

These features were reconstucted from the knn indices released by LAION using the code `tools/reconstruct_laion_features.py`. 

comment out following code in `faiss/IndexIVF.cpp` to reconstruct LAION features:
```cpp
void IndexIVF::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    // FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
```
