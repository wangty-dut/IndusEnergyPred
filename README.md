<div align="center">
  <h2><b> An Industrial Energy Prediction Method Integrating Planning Information and Process Correlation Characteristics </b></h2>
</div>

<div align="center">



</div>

<div align="center">


</div>

<p align="center">


## Introduction

An accurate prediction of energy production and consumption is a prerequisite for realizing reasonable energy 
scheduling in process industry. However, under the conditions of production-energy coupling, the energy operating
status is highly related to production rhythm. Many prediction methods are unable to consider the impact of 
multi-production process correlation and planning constraints on energy data fluctuations. To tackle this problem, 
an industrial energy prediction method integrating plan and multi-dimensional data correlation is proposed. 
A data augmentation method based on wavelet matching is developed to extract specific features of the energy data 
and obtain augmented samples. To capture the alternating operation characteristics of different production processes, 
a contrastive learning (CL) method with probability jumping is developed that takes the process uncertainty into 
consideration. On this basis, the planning information are represented in a novel form of partial differential 
equations (PDEs), so that the global production information can be embedded as a priori knowledge by a physics-informed
neural network (PINN) to achieve a dynamic energy prediction. In order to validate the effectiveness of the proposed
method, experiments are conducted using energy data from a steel company and compared with a variety of 
state-of-the-art methods. The results verify that the proposed method achieves better prediction results in complex 
industrial scenarios containing production process coupling and planning constraints.
<p align="center">
<img src="./figures/framework.png" height = "360" alt="" align=center />
</p>


## Requirements
Use python 3.8 from Conda
- matplotlib==3.7.5
- numpy==1.24.3
- pandas==2.0.3
- torch==2.3.0

