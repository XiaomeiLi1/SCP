# SCP
## Stable breast cancer prognosis
### Xiaomei Li<sup>1</sup>, Lin Liu<sup>1</sup>, Jiuyong Li<sup>1</sup>, Thuc Duy Le<sup>1, 2</sup>

1. UniSA STEM, University of South Australia, Mawson Lakes, SA, Australia
2. Centre for Cancer Biology, an alliance of SA Pathology and University of South Australia, Adelaide, SA, Australia

Breast cancer is the leading cause of cancer death for women and accounts for 30% of new female cancer cases in the United States in 2020. Breast cancer does not have a universal treatment because it is a complex disease with different types. Predicting breast cancer prognosis and stratifying patients into different risk groups can help tailor treatment for patients and improve their survival outcomes. Various computational methods using transcriptomic data have been proposed for breast cancer prognosis. However, these methods suffer from unstable performances on testing datasets which might be highly heterogeneous and follow the different distribution to the training dataset. Therefore, we propose a novel method, Deep Global Balancing Cox regression (DGBCox) , to improve breast cancer prognosis across multiple testing datasets.

This repository includes the scripts and data of the proposed method. 

The R folder includes:

- datasets.R - script for preprosses datasets
- benchmark.R - script for benchmark methods
- LogRank.R - script for Log rank test
- metrics.R - script for evaluation metrics
- Robust.R - script for Cox regreesion using robust features

The DGBCox folder includes:

- utils.py - Script for other support functions
- f_DGBCox.py - Script for the DGBCox method
- f_baselines.py - Script for the baseline methods: Coxnet, DAECox
- test.py - Script for testing new patients using DGBCox
- VIP.py - Script for the permutation importance method 
- visualization.py - Script for plotting the survival curves 

The data folder includes:
- TCGA500.csv - example test data
- TCGA753.csv - example test data


Notes:

(1) GEO736, GSE6532, UK, HEL, GSE19783 can be downloaded from the link(https://github.com/XiaomeiLi1/Datasets/releases/tag/V1.0.1). 

(2) The MAINZ, TRANSBIG, UPP, UNT, and NKI datasets are from Bioconductor (\url{https://bioconductor.org/}). Please install the following packages and include the datasets in your code:

```
library(breastCancerMAINZ)
library(breastCancerTRANSBIG)
library(breastCancerUPP)
library(breastCancerUNT)
library(breastCancerNKI)
```

(3) The METABRIC data need to be download from the EMBL-EBI repository (https://www.ebi.ac.uk/ega/, accession number EGAS00000000083, require individual access agreement).

(4) Please install CancerSubtypesPrognosis package from https://github.com/XiaomeiLi1/CancerSubtypesPrognosis.

(5) Please test DGBCox on your own data with the following code:

```
python test.py <path to your dataset>
```
