# Project Structure Overview

## FCGHUNTER

attack samples:
https://drive.google.com/file/d/1OWIWVVjifCv3iByRP4IBx-Sn8d49oB_4/view?usp=drive_link

target model:
https://drive.google.com/file/d/15HIjy9QIrwjOwzD_-pJxFb-uCMKSwPyP/view?usp=drive_link

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
