# Demographic bias in misdiagnosis by computational pathology models

*Nature Medicine* <img src=".github/cover.jpg" width="300px" align="right" />

[Journal Link](https://doi.org/10.1038/s41591-024-02885-z) | [Cite](#cite)

**Abstract:** Despite increasing numbers of regulatory approvals, deep learning-based computational pathology systems often overlook the impact of demographic factors on performance, potentially leading to biases. This concern is all the more important as computational pathology has leveraged large public datasets that underrepresent certain demographic groups. Using publicly available data from The Cancer Genome Atlas and the EBRAINS brain tumor atlas, as well as internal patient data, we show that whole-slide image classification models display marked performance disparities across different demographic groups when used to subtype breast and lung carcinomas and to predict _IDH1_ mutations in gliomas. For example, when using common modeling approaches, we observed performance gaps (in area under the receiver operating characteristic curve) between white and Black patients of 3.0% for breast cancer subtyping, 10.9% for lung cancer subtyping and 16.0% for *IDH1* mutation prediction in gliomas. We found that richer feature representations obtained from self-supervised vision foundation models reduce performance variations between groups. These representations provide improvements upon weaker models even when those weaker models are combined with state-of-the-art bias mitigation strategies and modeling choices. Nevertheless, self-supervised vision foundation models do not fully eliminate these discrepancies, highlighting the continuing need for bias mitigation efforts in computational pathology. Finally, we demonstrate that our results extend to other demographic factors beyond patient race. Given these findings, we encourage regulatory and policy agencies to integrate demographic-stratified evaluation into their assessment guidelines.


## Overview of repository
In this repository, we provide the code to reproduce our results of race-stratified and overall evaluation of *IDH1* mutation prediction in gliomas using Whole Slide Images (WSIs) and multiple instance learning (MIL). We train models on EBRAINS brain tumor atlas and test on TCGA-GBMLGG. Our study also includes subtyping of breast and lung carcinomas. However, since the evaluation of breast and lung subtyping tasks is based on private WSIs collected from Mass General Brigham, Boston, we are unable to make this data public.  

## Installation Guide for Linux (using anaconda)
### Prerequisites: 
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on Nvidia GeForce RTX 3090 Ti) with CUDA 11.7
- Python (version 3.8.13), PyTorch (version 2.0.0, CUDA 11.7), OpenSlide (version 4.3.1), openslide-python (version 1.2.0), Pillow (version 9.3.0), Scikit-learn (version 1.2.1), Matplotlib (version 3.7.1), Seaborn (version 0.12.2), Numpy (version 1.24.4), pandas (version 1.5.3), slideflow (version 2.1.0), smooth-topk.
- Code adapted from [CLAM](https://github.com/mahmoodlab/CLAM)

### Downloading TCGA and EBRAINS Brain Tumor Atlas Data
To download diagnostic WSIs (formatted as .svs files), molecular feature data and other clinical metadata, please refer  to the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov) and the [cBioPortal](https://www.cbioportal.org/). WSIs for each cancer type can be downloaded using the [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/). For EBRAINS Brain Tumor Atlas, data can be downloaded from [EBRAINS](https://search.kg.ebrains.eu/instances/Dataset/8fc108ab-e2b4-406f-8999-60269dc1f994). 

## Processing Whole Slide Images 
To process WSIs, first, the tissue regions in each biopsy slide are segmented using Otsu's Segmentation on a downsampled WSI using OpenSlide. Then, 256 x 256 patches without spatial overlapping are extracted from the segmented tissue regions at the desired magnification. We use three different feature encoders: $\text{ResNet50}$ trained on ImageNet ([ResNet50IN](https://github.com/mahmoodlab/CLAM)), Swin-T transformer trained on histology slides ([CTransPath](https://github.com/Xiyue-Wang/TransPath)), and a Vision Transformer Large trained on histology slides ([UNI](https://github.com/mahmoodlab/UNI)). We also extract Macenko stain normalized features using these feature extractors. We use the [CLAM](https://github.com/mahmoodlab/CLAM) toolkit to do all pre-processing.  

## Training-Validation splits on EBRAINS
`splits_MonteCarlo` contains the 20-fold label-stratified Monte Carlo splits for *IDH1* mutation prediction. The folder also contains the indepedent test cohort TCGA-GBMLGG and its associated metadata.

## Exploring modeling techniques
We experiment with various modeling choices for all components of the typical computational pathology pipeline:
1. `Patch feature extractor:` We use ResNet50IN, CTransPath, and UNI
2. `Patch aggregators:` We use ABMIL, CLAM, and TransMIL 
3. `Bias mitigation strategies:` We investigate Importance Weighting and Adversarial Regularization. Note: these techniques cannot be used for *IDH1* mutation prediction as patient race is not provided in EBRAINS.

You can find the implementations in the `models` directory. 

## Running Experiments 
Once you have segmented tissue, created patches, and extracted the features, you can refer to `commands` folder to train and evaluate models. Hyper-parameters used to train these models are directly taken from their respective papers. We provide commands for all combinations of patch feature extractors (ResNet50IN, CTransPath, UNI) and aggregators (ABMIL, CLAM, TransMIL) for *IDH1* mutation prediction. 

## Fairness criteria
After you have trained and evaluated your model, you can use `analysis/compute_TPR_disparity.py` to compute the race-startified (white, Asian, and Black patients) and overall performance metrics (AUC, F1) and fairness (TPR disparity) metrics. When prompted to provide `path/to/test/df`, you can use the path to the independent test set csv (i.e., `./splits_MonteCarlo/TCGA_GBMLGG/test.csv`). 

## Cite
If you find our work useful in your research or if you use parts of this code please consider citing our [paper](https://www.nature.com/articles/s41591-024-02885-z):

Vaidya, A., Chen, R.J., Williamson, D.F.K., et al. Demographic bias in misdiagnosis by computational pathology models. Nat Med (2024). [https://doi.org/10.1038/s41591-024-02885-z](https://www.nature.com/articles/s41591-024-02885-z)

```bibtext
@article{vaidya2024demographic,
  title={Demographic bias in misdiagnosis by computational pathology models},
  author={Vaidya, Anurag and Chen, Richard and Williamson, Drew and Song, Andrew and Jaume, Guillaume and Yang, Yuzhe and Hartvigsen, Thomas and Dyer, Emma and Lu, Ming and Lipkova, Jana and Shaban, Muhammad and Chen, Tiffany and Mahmood, Faisal},
  journal={Nature Medicine},
  publisher={Nature Publishing Group},
  year={2024}
}
```

## Issues 
- Please open new threads or report issues directly (for urgent blockers) to `avaidya@mit.edu`.
- Immediate response to minor issues may not be available.

## License and Usage 
[Mahmood Lab](https://faisal.ai) - This code is made available under the CC-BY-NC-ND 4.0 License and is available for non-commercial academic purposes.

![alt text](.github/logo.png)
