<p align="center">

  <h1 align="center">Enhancing Robustness Testing for Graph-based Android Malware Detection via Dependency-Aware Mutation and Multi-Objective Optimization</h1>
  <div>This repository contains the datasets, training scripts for target Android Malware Detection (AMD) models, and testing tools( FCGHUNTER, BagAmmo and HRAT) for evaluating AMD models used in this study.</div>
    <br>

</p>

---

### Dataset 

**1. Attack Samples**:
[Download Attack Samples](https://drive.google.com/file/d/1OWIWVVjifCv3iByRP4IBx-Sn8d49oB_4/view?usp=drive_link)  
This includes all graph representations of attack samples, which can be directly used as input for this project.

**2. Target Model**:
[Download Target Model](https://drive.google.com/file/d/15HIjy9QIrwjOwzD_-pJxFb-uCMKSwPyP/view?usp=drive_link)  
Contains 40 target models spanning 8 feature types and 5 classifiers.

**3. Surrogate Model**:
[Download Surrogate Model](https://drive.google.com/file/d/1pyCCWTCH9XtuaLNbnGsTeAb4Nv1_A4K8/view?usp=drive_link)  
Includes KNN surrogate models across 8 feature types and a GCN model for BagAmmo. Additionally, KNN's benign and malware models are included to compute KNN distances.

**4. Training and Test Features**:
[Download Training & Test Features](https://drive.google.com/file/d/1AjNfQw7Z2Vom8KPpqfO6nHKimE44pEc4/view?usp=drive_link)  
For users who wish to train models themselves.


### Execute Attack

- **GA Framework**:
  - `main_attack/MutateFCG_[feature_type]_[target_model].py`: This script implements a Genetic Algorithm (GA) framework tailored for different feature types and target models.

### Util Classes

- **FCG Initialization and Operations**:
  - `type/FCG.py`: Initializes FCG (Function Call Graph), includes various perturbation operators about FCG, and processes operator subsequences.
- **Mutation Operations**:
  - `Type/Mutation.py`: Packages some basic atomic operations about FCG.
- **Dependency-aware Sequence Construction**:
  - `Type/MutationSequence.py`: Constructs the dependency-aware sequence to manage operation dependencies effectively.

## Other Utilities

- **Classification**:
  - `main_attack/Classify.py`: Defines, trains, and tests all target models, including 5 ML-based classifiers and GCN/MLP surrogate models.
- **FCG Extraction**:
  - `main_attack/ExtractFCG.py`: Extracts FCG from an APK and stores it in a NetworkX GEXF format.
- **Feature Extraction**:
  - `main_attack/ExtractFeature.py`: Extracts feature vectors from FCG, including three types of graph embeddings.

## Baselines

### Baseline 1: BagAmmo

- **Execute Attack**:
  - For example, for `feature_type: APIGraph, target_model: MLP`, execute:
    - `main_attack/baseline_APIGraph_add_edge_MLP`
  - Parameters are specified within the files.

### Baseline 2: HRAT

- **Execute Attack**:
  - `HRAT-copy-main/HRAT_Attack_Malscan/main_attack_[feature_type]_[target_model]`

## Repackaging: Java Project

1. **Preparation**:
   - Place the generated sequences and original APK in the designated folder.

2. **Execute Repackaging**:
   - Run `Repackage/APPMod/FCGModification/src/ZacharyKz/mainModificationNew.java` to apply the modifications to the APK.
