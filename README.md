# Application of Neural Collaborative Filtering & Deep Matrix Factorization on Drug Target Interaction
This is code for the course project for the Collaboarative Filtering(CSE640) Fall'18 course offered in IIIT Delhi.

## Introduction
We build a recommender system for drug target interactions using two warm start methods: Neural Collaborative Filtering and Deep Matrix Factorization. We apply these methods to the drug target interaction problem to predict what kind of
drugs would work on which target site.

## Navigating the code
- `drug.py` loads the datasets.
- `ncf.py`, `dmf.py`, `GMF.py`, `MLP.py` contains code for training and saving checkpoints for the DMF and NCF models.
- `evaluate.py` is the evaluation script for computing the NDCG and Hit Ratio metrics on the saved models.
- The results obtained with the trained models are shown in `dmf_results` and `ncf_results`.

Note: Code adapted from [here](https://github.com/RuidongZ/Deep_Matrix_Factorization_Models) and [here](https://github.com/hexiangnan/neural_collaborative_filtering/)
