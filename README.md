# PowerGuard AI — SmartGrid Intelligence Platform

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ML Accuracy](https://img.shields.io/badge/ML%20Accuracy-95.8%25-green.svg)](#high-fidelity-ml-architecture)

PowerGuard AI is an enterprise-grade SmartGrid intelligence platform designed to detect electricity theft and meter faults using a High-Fidelity Machine Learning Ensemble. By blending real-world smart meter profiles (CEEW Bareilly 2020) with sophisticated stochastic noise, the platform achieves realistic, industrially viable detection rates.

## 🚀 Key Features

- **High-Fidelity ML Ensemble**: Combines Random Forest, Gradient Boosting, SVM, and KNN for robust classification.
- **Explainable AI Integration**: Per-meter risk scoring with detailed action steps for field teams.
- **Enterprise Dashboard**: Premium Glassmorphic interface featuring real-time anomaly feeds and PCA feature space visualization.
- **Dynamic Anomaly Injection**: Models real-world grid complexities including label flip noise and feature range crossover.

## 🛠️ Tech Stack

- **Backend**: Python, Flask, Flask-CORS
- **Intelligence**: Scikit-Learn, NumPy, Pandas
- **Frontend**: HTML5, Vanilla CSS (Premium Aesthetics), Chart.js
- **Data Source**: CEEW Bareilly 2020 Smart Meter Dataset (Hybrid Baseline)

## 📊 Dataset Overview

The platform includes a processed **High-Fidelity Hybrid Dataset** located in `data/powerguard_hybrid_dataset_v2.5.csv`. 
- **Samples**: 25,000 processed meter readings.
- **Features**: 96 dimensions (10 numeric + 6 PCA + 50 NLP-based complaint features + 30 TF-IDF features).
- **Stochasticity**: Includes intentional 6% label noise and 15% profile blurring to match real-world audit blind spots.

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/powerguard-ai.git
   cd powerguard-ai
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Backend**:
   ```bash
   python app.py
   ```

5. **Launch the Dashboard**:
   Open `index.html` in any modern browser.




## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---
*Developed for advanced grid intelligence and revenue protection.*
