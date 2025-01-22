## Description
The VLS2ketch datasets and source code are intended for Sketching Very Large-scale Dynamic Attributed Networks More Practically.

## Install Intel MKL
Intel MKL is used for basic linear algebra operations.
You can download the mkl library online according to the official website
https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=linux&distributions=offline

Remember the intel installation path {intel_PATH}

# Compile

To compile VLS2ketch, you may need to edit Makefile when you install MKL. You need to set something like:
```
INCLUDE_DIRS = -I./ligra -I./pbbslib -I./mklfreigs -I"{ANACONDA_PATH}/envs/VLS2ketch/include" -I"{intel_PATH}/oneapi/mkl/2023.2.0/include"
LINK_DIRS = -L"{ANACONDA_PATH}/envs/VLS2ketch/lib" -L"{intel_PATH}/oneapi/mkl/2023.2.0/lib/intel64"
```
To clean the compiled file, run `make clean`.
Then run `make` to compile.

## Data download
The download [link](https://v50tome-my.sharepoint.com/:f:/g/personal/vls2ketch_v50tome_onmicrosoft_com/EtNVPqCgbNZAuDm_LGkEEy8BJXq5JI7POnzNN7KSwXRUQg?e=jUdXtM) for the dynamic dataset ogbn-product.
The download [link](https://v50tome-my.sharepoint.com/:f:/g/personal/vls2ketch_v50tome_onmicrosoft_com/EpIIdz_ZyMhHtsN4F-Rw34gBK8EWajuLJfs3szFwsUvVPA?e=cwqwsz) for the dynamic dataset MAG-Scholar-C.
The download [link]() for the dynamic dataset ogbn-papers100M.
```
data
├── ogbn-product       
├── MAG-Scholar-C  
├── ogbn-papers100M  
```
## Data preprocessing
```
cd VLS2ketch
python dataPreProcess.py --dataset ogbn-product --timestep 10 --start_year 0
python SparseRandomMatrix.py --dataset ogbn-product --K 200 --timestep 10 --start_year 0
```

## Run the VLS2ketch model
```
bash VLS2ketch.sh
```

## Get the embedding in python
```
python getEmbedding.py --dataset ogbn-product --K 200 --timestep 10 --start_year 0
```

## Evaluation
```
python nodeClassification.py --dataset ogbn-product --K 200 --timestep 10 --start_year 0     
python linkPrediction.py --dataset ogbn-product --K 200 --timestep 10 --start_year 0   
```

