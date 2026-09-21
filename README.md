# Privacy-Preserving Federated Learning for Glaucoma Detection

## Overview

This project explores the use of **Federated Learning (FL)** for glaucoma detection across multiple simulated healthcare institutions, while keeping patient retinal images local to each institution.

Instead of collecting fundus images from different hospitals into a centralized location, each simulated hospital trains a local model on its own data. Only model updates — not raw images — are shared with a central server, where they are aggregated into an improved global model. This lets multiple institutions contribute to a shared model without directly sharing patient data.

The project specifically investigates how **heterogeneous (non-IID) data distributions** across hospitals — for example, one hospital seeing mostly healthy patients and another seeing mostly glaucoma cases — affect federated learning performance compared to centralized training.

**Research question:** How does variation in data distribution and image quality between simulated healthcare institutions affect the performance of federated learning for glaucoma detection, compared with centralized learning — and how much extra privacy/security protection can be added before it starts costing too much accuracy?

## Objectives

* Implement a federated learning workflow for glaucoma detection using retinal fundus images.
* Simulate three healthcare institutions ("hospitals") with separate, unequal datasets.
* Train a ResNet-18 model locally at each simulated hospital via transfer learning.
* Aggregate local model updates using **FedAvg** and compare against **FedProx**.
* Compare federated learning against centralized training as a baseline.
* Investigate the effect of heterogeneous class distributions (normal vs. glaucoma) across hospitals on global model performance.
* Compare **secure aggregation** and **differential privacy** as optional add-on protections, and evaluate their accuracy trade-offs.
* Build a simple hospital login dashboard and deploy the system to AWS.

## Architecture

The project follows a simulated multi-hospital federated learning architecture:

```text
                ┌─────────────────────┐
                │   Federated Server   │
                │                     │
                │   Global Model      │
                │   FedAvg / FedProx  │
                └──────────┬──────────┘
                           │
             Model Updates │
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  Hospital A   │  │  Hospital B   │  │  Hospital C   │
│               │  │               │  │               │
│ Local Data    │  │ Local Data    │  │ Local Data    │
│ Local Model   │  │ Local Model   │  │ Local Model   │
│ (ResNet-18)   │  │ (ResNet-18)   │  │ (ResNet-18)   │
└───────────────┘  └───────────────┘  └───────────────┘
```

Raw fundus images remain within each simulated hospital. Only model parameters/updates are communicated to the federated server. Each hospital also has its own login-protected dashboard for managing its local data and training.

## Dataset

The project uses the **REFUGE dataset** (Retinal Fundus Glaucoma Challenge), accessed via a public Kaggle mirror ([kaggle.com/datasets/arnavjain1/glaucoma-datasets](https://www.kaggle.com/datasets/arnavjain1/glaucoma-datasets)), since direct registration with the official REFUGE organizers (refuge.grand-challenge.org) was declined.

REFUGE contains:
* 1,200 color retinal fundus photographs, split into train (400 images), validation, and test sets
* Glaucoma classification labels (`0` = normal, `1` = glaucoma) — confirmed via each split's `index.json`
* Optic disc and optic cup segmentation masks (`Masks` / `Masks_Cropped`), which could later support cup-to-disc ratio analysis
* Fovea coordinates (not used in this project)

The training set's class distribution (40 glaucoma / 360 normal) was verified to match the official REFUGE statistics, confirming the mirror is a faithful copy.

For the federated experiments, the training data is split across three simulated hospitals with deliberately different class balances:

| Hospital | Normal | Glaucoma |
|---|---|---|
| Hospital A | ~80% | ~20% |
| Hospital B | ~50% | ~50% |
| Hospital C | ~20% | ~80% |

An even/balanced 3-way split is also used as a control, to isolate the effect of heterogeneity specifically.

> **Note:** Raw dataset files are not included in this repository. See the official REFUGE source (refuge.grand-challenge.org) or the Kaggle mirror linked above for access and licensing information.

## Federated Learning Process

1. Preprocess the REFUGE fundus images (resize to 224×224, ImageNet normalization).
2. Divide the training data among the three simulated hospital clients (balanced, then heterogeneous).
3. Initialize a global ResNet-18 model (pretrained on ImageNet, final layer adapted for binary classification).
4. Send the global model to participating hospital clients.
5. Train the model locally on each hospital's data.
6. Send the resulting model updates to the federated server.
7. Aggregate the updates using **FedAvg**, then separately using **FedProx**, for comparison.
8. Update the global model.
9. Repeat for multiple communication rounds.
10. Evaluate the resulting global model against the centralized baseline.

## Technologies

* **Python** — Primary programming language
* **PyTorch** / **torchvision** — Deep learning framework and pretrained ResNet-18
* **Flower** — Federated learning framework
* **NumPy** / **Pandas** — Numerical computing and data processing
* **Matplotlib** — Visualization
* **Jupyter Notebook** — Experimentation and analysis
* **AWS (EC2, S3, IAM)** — Cloud deployment of the federated system

## Project Structure

```text
Glaucoma_FL_Project/
│
├── .venv/
│
├── data/
│   ├── raw/
│   │   └── REFUGE/          # train / val / test, Images_Cropped, Masks_Cropped, index.json
│   ├── processed/
│   └── hospitals/           # per-hospital data splits (balanced + heterogeneous)
│
├── models/
│
├── notebooks/
│
├── results/
│
├── src/
│   ├── preprocessing.py     # RefugeDataset class, transforms
│   ├── model.py              # ResNet-18 setup (planned)
│   ├── train.py               # centralized baseline training (planned)
│   ├── client.py              # Flower client (planned)
│   └── server.py              # Flower server / FedAvg / FedProx (planned)
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Evaluation

Model performance is evaluated using:

* Accuracy, Sensitivity (Recall), Specificity, Precision, F1-score, ROC-AUC
* Confusion matrix
* Training and validation loss curves
* Federated-specific metrics: number of communication rounds to convergence, per-client local accuracy, communication cost

Results are used to compare centralized training, FedAvg, and FedProx, both under balanced and heterogeneous hospital data.

## Privacy & Security Considerations

Federated learning reduces the need to move sensitive healthcare data to a centralized location, since training occurs locally at each hospital and only model updates are shared.

However, FL alone does not guarantee complete privacy — model updates can still leak information about the underlying training data (e.g., via gradient inversion or membership-inference attacks). This project plans to evaluate two additional protections as a comparative experiment:

* **Secure aggregation** — hides each hospital's individual update from the server, so only the combined average is ever visible.
* **Differential privacy (DP)** — adds calibrated noise to each hospital's update, providing a mathematically provable privacy guarantee at some cost to accuracy.

The planned experiment compares four setups — no privacy add-ons, secure aggregation alone, DP alone, and both combined — to characterize the accuracy/privacy trade-off rather than declare a single "best" method.

## Future Improvements

* Grad-CAM explainability (visualizing which image regions drive predictions)
* Cup-to-disc ratio analysis using REFUGE's segmentation annotations
* More realistic non-IID splits (e.g., simulating camera/equipment differences, not just class imbalance)
* Newer federated learning algorithms beyond FedAvg/FedProx (e.g., FedNova, SCAFFOLD)
* Blockchain-based update logging
* Multimodal data (combining OCT, fundus, and EHR data)
* Testing with real hospital partners (would require ethics/regulatory approval, out of scope for this project)

## Disclaimer

This project is intended for research and educational purposes only. It is **not a clinical diagnostic system** and should not be used to make medical decisions.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more information.
