# 🔬 An Industrial Energy Prediction Method Integrating Planning Information and Process Correlation Characteristics
This repository is the code implementation of the paper **"An Industrial Energy Prediction Method Integrating Planning Information and Process Correlation Characteristics"**.
## Project Introduction
This project includes experimental code for two datasets, **BFG** and **LDG**, which are stored in separate folders.   
The experiment mainly includes the following modules:
- **data augmentation**  
- **comparative learning**  
- **PINN training**  
- **test**  
- **refactor.m**  

## file structure
```plaintext
project_root/
│── BFG/ # BFG Experimental code
│ ├── data_pretreatment.py # data augmentation for feature construction and extraction
│ ├── contrastive_train.py # comparative learning with jumping module
│ ├── pinn_train.py # PINN training
│ ├── test.py # test
│ ├── refactor.m # Refactor data from the augmented features
│
│── LDG/ # LDG Experimental code
│ ├── data_pretreatment.py
│ ├── contrastive_train.py
│ ├── pinn_train.py
│ ├── test.py
│ ├── refactor.m # Refactor data
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
Taking the  **BFG experiment** as an example, the running sequence is as follows:

1.Comparative Learning Network Training
```bash
python BFG/contrastive_train.py
```
2.PINN network training
```bash
python BFG/pinn_train.py
```
3.test
```bash
python BFG/test.py
```
4.Data reconstruction and result plotting based on Matlab
```bash
BFG/refactor.m
```

The LDG experiment is the same, just run the corresponding script in the ```LDG/``` folder.

## Contact Information
If you have any questions about the code or experiment, please contact
Email:  wangty@dlut.edu.cn. Thanks!
