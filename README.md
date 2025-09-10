An Industrial Energy Prediction Method Integrating Planning Information and Process Correlation Characteristics
Introduction
An accurate prediction of energy production and consumption is a prerequisite for realizing reasonable energy scheduling in process industry. However, under the conditions of production-energy coupling, the energy operating status is highly related to production rhythm. Many prediction methods are unable to consider the impact of multi-production process correlation and planning constraints on energy data fluctuations. To tackle this problem, an industrial energy prediction method integrating plan and multi-dimensional data correlation is proposed. A data augmentation method based on wavelet matching is developed to extract specific features of the energy data and obtain augmented samples. To capture the alternating operation characteristics of different production processes, a contrastive learning (CL) method with probability jumping is developed that takes the process uncertainty into consideration. On this basis, the planning information are represented in a novel form of partial differential equations (PDEs), so that the global production information can be embedded as a priori knowledge by a physics-informed neural network (PINN) to achieve a dynamic energy prediction. In order to validate the effectiveness of the proposed method, experiments are conducted using energy data from a steel company and compared with a variety of state-of-the-art methods. The results verify that the proposed method achieves better prediction results in complex industrial scenarios containing production process coupling and planning constraints.
Requirements
arch==7.2.0
einops==0.8.1
fire==0.7.1
gin==0.1.006
joblib==1.4.2
keract==4.5.2
mamba_ssm==2.2.5
matplotlib==3.7.5
minepy==1.2.6
nltk==3.9.1
numpy==1.24.3
pandas==2.0.3
reformer_pytorch==1.4.4
scikit_learn==1.3.2
scipy==1.16.1
setuptools==69.5.1
sktime==0.38.5
statsmodels==0.14.5
sympy==1.13.2
tensorflow==2.13.0
tensorflow_intel==2.13.0
tqdm==4.66.5
wfdb==4.3.0
