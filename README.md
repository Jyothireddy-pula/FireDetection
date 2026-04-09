# 🔥 FireGuard AI: Precision Forest Fire Detection & Monitoring

FireGuard AI is a high-precision forest fire risk prediction system that combines **Machine Learning**, **Nature-Inspired Optimization**, and **Interpretability tools**. By leveraging the Algerian Forest Fires dataset and an optimized Neural Network, the system achieves near-perfect classification accuracy to support early warning systems and forestry management.

## 🚀 Key Features
- **$\approx 98\%$ Prediction Accuracy**: Optimized using Particle Swarm Optimization (PSO).
- **AI Explainability**: Integrated **SHAP** values to understand *why* the model predicts high risk.
- **Regional Monitoring**: A simulated 5x5 regional grid scanner for spatial risk assessment.
- **Sensitivity Analysis**: Trend simulators to analyze how weather variables (Temp, RH, Wind) impact fire probability.
- **Interactive Dashboard**: A professional Streamlit web interface for real-time risk prediction.

---

## 🛠️ Technical Architecture

### 1. The Pipeline (Phases 1-8)
The project is structured into 8 sequential phases:
- **Phase 1: Data Engineering** $\rightarrow$ Processing Algerian dataset, Min-Max scaling, and feature interaction engineering.
- **Phase 2: Fuzzy Sugeno Baseline** $\rightarrow$ Manual expert system with Gaussian membership functions.
- **Phase 3: Standard MLP** $\rightarrow$ Baseline Multi-Layer Perceptron regularization.
- **Phase 4: PSO Optimization** $\rightarrow$ Using Particle Swarm Optimization to find the optimal neuron count and learning rate.
- **Phase 5: Final Evaluation** $\rightarrow$ Comparative analysis and performance metrics.
- **Phase 6: SHAP Explainability** $\rightarrow$ Global and local feature importance analysis.
- **Phase 7: Regional Grid Scanner** $\rightarrow$ Batch processing of regional coordinates to create risk heatmaps.
- **Phase 8: Trend Simulator** $\rightarrow$ Parameter sensitivity analysis for predictive "what-if" scenarios.

### 2. Model Specifications
- **Algorithm:** Multi-Layer Perceptron (MLP)
- **Optimizer:** Particle Swarm Optimization (PSO)
- **Dataset:** Algerian Forest Fires Dataset
- **Optimal Architecture:** 13 $\to$ 10 neurons (determined by PSO)
- **Key Features:** Temperature, Relative Humidity, Wind Speed, FWI, and custom Interaction Indices.

---

## 💻 Installation & Usage

### Prerequisites
- Python 3.10+
- Virtual Environment (venv)

### Setup
1. **Clone the repository** and navigate to the project folder.
2. **Activate the virtual environment**:
   ```bash
   # Windows
   .\\venv\\Scripts\\activate
   # Linux/Mac
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline
To generate the models and all documentation plots from scratch:
```bash
python phase1_data.py && python phase2_fuzzy.py && python phase3_mlp.py && python phase4_pso.py && python phase5_results.py && python phase6_shap.py && python phase7_grid.py && python phase8_simulator.py
```

### Launching the Dashboard
To start the interactive web interface:
```bash
python -m streamlit run app.py
```

---

## 📊 Results Summary
| System | Accuracy | F1-Score |
| :--- | :---: | :---: |
| Plain Fuzzy Sugeno | 65.31% | 0.5854 |
| Standard MLP | 97.96% | 0.9796 |
| **PSO-MLP (Final)** | **97.96%** | **0.9796** |

---

## 📁 Project Structure
- `/models` - Saved `.pkl` models and `.npy` datasets.
- `/outputs` - All generated plots, heatmaps, and result tables.
- `app.py` - The Streamlit frontend application.
- `phase1_data.py` $\dots$ `phase8_simulator.py` - The complete research pipeline.
- `archive/` - Raw dataset storage.

## 📜 References
- Algerian Forest Fires Dataset.
- SHAP (SHapley Additive exPlanations) library.
- Scikit-learn MLP & MinMaxScaler.
- Particle Swarm Optimization (PSO) research.
