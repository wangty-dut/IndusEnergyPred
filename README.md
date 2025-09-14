# 🔬 An Industrial Energy Prediction Method Integrating Planning Information and Process Correlation Characteristics
This repository is the code implementation of the paper **"An Industrial Energy Prediction Method Integrating Planning Information and Process Correlation Characteristics"**.
## Project Introduction
This project includes experimental code for two datasets, **BFG** and **LDG**, which are stored in separate folders.   
The experiment mainly includes the following modules:
- **data augmentation**   (Includes feature enhancement Matlab code and Python code)
- **contrastive learning**  
- **PINN training**  
- **test**  
- **refactor.m**  

## file structure
```plaintext
project_root/
│── BFG/ # BFG Experimental code
│ ├── Extract_rhythm.m # Extract rhythm information
│ ├── Extract_waveform_features.m # Extract waveform data information
│ ├── feature_merge.m # feature integration
│ ├── data_pretreatment.py
│ ├── contrastive_train.py # contrastive learning with jumping module
│ ├── pinn_train.py # PINN training
│ ├── test.py # model testing
│ ├── refactor.m # Refactor data based on the augmented sfeatures
│
│── LDG/ # LDG Experimental code
│ ├── get_feature.py # Extract rhythm information
│ ├── data_pretreatment.py
│ ├── contrastive_train.py
│ ├── pinn_train.py
│ ├── test.py
│ ├── refactor.m
│
│── README.md
└── requirements.txt
```

## Environmental Requirements
- Python 3.8
- matplotlib==3.7.5
- numpy==1.24.3
- pandas==2.0.3
- torch==2.3.0

Other dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```

## run steps
**BFG experiment**

1.Data feature enhancement processing
```bash
BFG/Extract_rhythm.m
BFG/Extract_waveform_features.m
BFG/feature_merge.m
```
2.Contrastive learning network training
```bash
python BFG/contrastive_train.py
```
3.PINN network training
```bash
python BFG/pinn_train.py
```
4.Model testing
```bash
python BFG/test.py
```
5.Data reconstruction and plotting based on Matlab
```bash
BFG/refactor.m
```

**LDG experiment**

1.Data feature enhancement processing
```bash
LDG/get_feature.py
```
2.The remaining steps of the LDG experiment are the same as those of the BFG experiment, just run the corresponding script in the ```LDG/``` folder.

## Contact Information
If there are any questions about the codes and datasets, please don't hesitate to contact us. Thanks!
