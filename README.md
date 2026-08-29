
[<img src="https://img.shields.io/badge/arXiv-/2606.25456-b31b1b"></img>](https://arxiv.org/abs/2606.25456)
# Towards Robust EEG Decoding Based on Riemannian Self-Attention


# README
This is the official code for our KDD 2026 publication: Towards Robust EEG Decoding Based on Riemannian Self-Attention.

## Experiments

The implementation is based on the official code of 
    
- *MAtt: A manifold attention network for EEG decoding* [[Neurips 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c981fd12b1d5703f19bd8289da9fc996-Abstract-Conference.html)] [[code](https://proceedings.neurips.cc/paper_files/paper/2022/hash/c981fd12b1d5703f19bd8289da9fc996-Supplemental.zip)].



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


## Reference
```bash
@article{pan2022matt,
  title={MAtt: A manifold attention network for EEG decoding},
  author={Pan, Yue-Ting and Chou, Jing-Lun and Wei, Chun-Shu},
  journal={NeurIPS},
  pages={31116--31129},
  year={2022}
}
```
