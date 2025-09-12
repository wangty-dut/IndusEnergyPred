# 🔬 An Industrial Energy Prediction Method Integrating Planning Information and Process Correlation Characteristics
This repository is the code implementation of the paper **"An Industrial Energy Prediction Method Integrating Planning Information and Process Correlation Characteristics"**.
## Project Introduction
This project includes experimental code for two sets of data, **BFG** and **LDG**, which are stored in separate folders.   
The experiment mainly includes the following modules:
- **data augmentation**  
- **Comparative learning and training**  
- **PINN training**  
- **test**  

## file structure
```plaintext
project_root/
│── BFG/ # BFG Experimental code
│ ├── data_pretreatment.py # data augmentation
│ ├── contrastive_train.py # Comparative learning and training
│ ├── pinn_train.py # PINN training
│ ├── test.py # test
│
│── LDG/ # LDG Experimental code
│ ├── data_pretreatment.py
│ ├── contrastive_train.py
│ ├── pinn_train.py
│ ├── test.py
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
The LDG experiment is the same, just run the corresponding script in the ```LDG/``` folder.

## Contact Information
If you have any questions about the code or experiment, please contact
Email:  wangty@dlut.edu.cn. Thanks!
