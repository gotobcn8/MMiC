# MMiC: Mitigating Modality Incompleteness in Clustered Federated Learning
**This work has accepted by CIKM'2025 [[Link](https://arxiv.org/pdf/2505.06911)].**

A framewrk for mitigating missing modality

## Dataset Preparation

#### [CrisisMMD](https://crisisnlp.qcri.org/crisismmd)

(1) Download V2.0

(2) Make the repository like:

```bash
repository/crisis_mmd/raw/CrisisMMD_v2.0/
├── annotations
├── crisismmd_datasplit_all
├── crisismmd_datasplit_all.zip
├── data_image
├── json
└── __MACOSX
```

Finished.

#### [Food101](https://www.kaggle.com/datasets/gianmarco96/upmcfood101/data)

```bash
repository/food101/raw/food101/
├── images
│   ├── test
│   └── train
└── texts
    ├── test_titles.csv
    └── train_titles.csv
```



## Run MMiC

Envrionment Setup

```bash
conda env create -n mmic python==3.9
```

Python version greater than 3.8 is available.

```bash
pip install -r requirements.txt
```



Run it!

```bash
python main.py -f config/multimodal/complete.yaml 
```



Structures of this framework

## Structure

- clients
  - Multimodal
    - mmclient.py
  - client.py (client base)
  - [fl algorithm].py (client in different algorithm)
- cluster
  - clusterbase.py
- dataset
  - multimdal
    - crissi_mmd.py
  - collector
    - tools.py(seperating dataset)
  - agnews.py (dataset provided)
  - download.py (download dataset)
- models (initialize models in clients respectively)
  - multimdal
    - pretrain_clip.py
- repository (For storage only)
- servers
  - multimdal
    - multimodalserver.py
  - serverbase.py (server base)
  - serverapi.py (execute entrance)
- utils (utils package)
- config
  - multimodal
    - mmic-food101.yaml
- algorithm (Some equations)



```
@article{yang2025mmic,
  title={MMiC: Mitigating Modality Incompleteness in Clustered Federated Learning},
  author={Yang, Lishan and Zhang, Wei and Sheng, Quan Z and Chen, Weitong and Yao, Lina and Shakeri, Ali},
  journal={arXiv preprint arXiv:2505.06911},
  year={2025}
}
```

