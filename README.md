<p align="center">

  <h1 align="center">Enhancing Robustness Testing for Graph-based Android Malware Detection via Dependency-Aware Mutation and Multi-Objective Optimization</h1>
  <div>This repository contains the datasets, training scripts for target Android Malware Detection (AMD) models, and testing tools( FCGHUNTER, BagAmmo and HRAT) for evaluating AMD models used in this study.</div>
    <br>

</p>

---

### 1. Dataset 

**(1). Attack Samples**:
[Download Attack Samples](https://drive.google.com/file/d/1OWIWVVjifCv3iByRP4IBx-Sn8d49oB_4/view?usp=drive_link)  
This includes all graph representations of attack samples, which can be directly used as input for this project.

**(2). Target Model**:
[Download Target Model](https://drive.google.com/file/d/15HIjy9QIrwjOwzD_-pJxFb-uCMKSwPyP/view?usp=drive_link)  
Contains 40 target models spanning 8 feature types and 5 classifiers.

**(3). Surrogate Model**:
[Download Surrogate Model](https://drive.google.com/file/d/1pyCCWTCH9XtuaLNbnGsTeAb4Nv1_A4K8/view?usp=drive_link)  
Includes KNN surrogate models across 8 feature types and a GCN model for BagAmmo. Additionally, KNN's benign and malware models are included to compute KNN distances.

**(4). Training and Test Features**:
[Download Training & Test Features](https://drive.google.com/file/d/1AjNfQw7Z2Vom8KPpqfO6nHKimE44pEc4/view?usp=drive_link)  
For users who wish to train models themselves.

### 2. Enviroment

Please downloading the corresponding dataset in advance, then installing the necessary libraries. 

```
python install -r requirements.txt
```


### 3. Execute Attack

##Introdcution about the structure

Our codebase is organized as follows:

- **FCGHUNTER**
- **BagAmmo**
- **HRAT**

Additionally, there is an "Other" directory which includes extra tasks:

- **Classify**: Contains scripts for training target models and substitute models (e.g., MLP, GCN).
- **ExtractFCG**: Extracts function call graphs (FCG) from APK files.
- **ExtractFeature**: Includes all feature extraction methods.

The "type" directory contains utility classes:

- **FCG**: Initializes the Function Call Graph (FCG), includes various perturbation operators, and processes operator subsequences.
- **Mutation**: Provides basic atomic operations related to FCG mutations.
- **MutationSequence**: Constructs dependency-aware sequences to manage operation dependencies.

The `utils.py` script includes basic functions used during the attack, such as file saving and logging.

#### FCGHUNTER

1. Example of execution:

#### BagAmmo

1. Example of execution:

#### HRAT

1. Example of execution:

