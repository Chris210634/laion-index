# Implementation of Paried k-means Indexing
----------

## Step 1 

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

## Step 2 



