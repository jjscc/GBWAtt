
[<img src="https://img.shields.io/badge/arXiv-/2606.25456-b31b1b"></img>](https://arxiv.org/abs/2606.25456)
# Towards Robust EEG Decoding Based on Riemannian Self-Attention


# README
This is the official code for our KDD 2026 publication: Towards Robust EEG Decoding Based on Riemannian Self-Attention.


### Dataset
We further release our preprocessed datasets. 


Please download (https://drive.google.com/file/d/1T6ay9KKzhgM1hg05w8Buefok58MMevYh/view?usp=drive_link) data and put it in the folder './GBWAtt/data'


### Running experiments
To train and test the experiments on the BCIC-IV-2a, MAMEM-SSVEP-II and BCI-ERN datasets, run this command:

```train and test
python GBWAttBCI.py
python GBWAttMamem.py
python GBWAttBciCha.py
```
