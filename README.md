# 🧬 PrediLnc

**PrediLnc** is a standalone software and web-based platform for predicting top **lncRNA–disease associations** using advanced biological feature processing and machine learning. Given an input **lncRNA** or **disease**, along with associated sequences or target data, PrediLnc uses our custom model **GARNet (Graph convolution Attention RNA Network)** to return the most relevant predictions. It serves as a powerful tool for hypothesis generation in biomedical research.

---

## 🧠 Abstract

Long non-coding RNAs (lncRNAs), transcripts >200 nucleotides, regulate gene expression, chromatin remodeling, and other key cellular processes. Dysregulation of lncRNAs is linked to a wide range of diseases like cancer, neurodegenerative, and cardiovascular conditions. Understanding their associations with diseases can reveal novel diagnostic and therapeutic avenues.

We present **PrediLnc**, built on **GARNet**, a predictive model that integrates:

- **Autoencoders** for dimensionality reduction  
- **Graph Convolutional Networks (GCNs)** for biomolecular topology modeling  
- **Self-attention mechanisms** for contextual awareness  
- **Stacked ensemble learning** including:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - Gradient Boosting
  - Multi-Layer Perceptron (MLP)
  - A Random Forest meta-classifier

GARNet was validated using stratified cross-validation and ten real-world case studies, with predictions supported by over 500 PubMed articles. The software is available both as a live demo and for local deployment.

---

## 🌐 Live Demo

👉 [Try the Web App](http://predilnc.dhanjal-lab.iiitd.edu.in/)

---
## ⚙️ Functionalities

### 🔬 Predict Diseases for a lncRNA
- Click on the **"Diseases for a lncRNA"** button.
- Select a lncRNA from the dropdown list.
- If your lncRNA is **not listed**, please provide:
  - lncRNA **sequence**
  - Related **diseases**
  - Associated **target genes**
- If your lncRNA **is listed**, you can directly view the predicted disease associations along with confidence scores based on our pre-processed data.

### 🧬 Predict lncRNAs for a Disease
- Click on the **"lncRNAs for a Disease"** button.
- Enter the **disease name** and optional **related genes**.
- The system will return the top-ranked lncRNAs associated with that disease.

### 📚 Additional Sections
- **About the Features**: Explore the biological features used in predictions.
- **Insights**: Learn about our workflow and potential applications.
- **Contribute**: Submit new data to improve the platform.
- **Contact**: Get details about authors and contributors.

---

## 🚀 Installation and Setup (Local Deployment)

### 🔧 Prerequisites

- Anaconda or Miniconda  
- Python 3.10  
- R ≥ 4.1.2  
- Redis 7.2.1  
- Git  
- Internet (for Zenodo model download)

---

### 📥 Step-by-Step Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/Udit64/PrediLnc.git
cd PrediLnc

# 2. Install Anaconda or Miniconda (manual step – download from official site)
#    Anaconda: https://www.anaconda.com/products/distribution
#    Miniconda: https://docs.conda.io/en/latest/miniconda.html

# 3. Create and activate a new conda environment with Python and R
conda create -n rtest python=3.10 r-base=4.1.2 rpy2 r-essentials -c conda-forge
conda activate rtest

# 4. Install required Python packages
pip install -r requirements.txt

# 5. Install Redis 7.2.1
git clone https://github.com/redis/redis.git
cd redis
git checkout 7.2.1
make -j
src/redis-server &       # Run Redis in the background

# To check Redis is running correctly
redis-cli ping            # Should return: PONG
cd ..                     # Return to PrediLnc root

# 6. Download pretrained models and data from Zenodo
zenodo_get --access-token G46P8DtW8lfKSG0u7IUVmCMb4idEKAoDCBL1yHWuwUkKvnFuGPSNCIkCham2 15764921

# 7. Extract dataset contents to app directory
unzip saved_dataset.zip -d app/

# Ensure app/ contains:
# - models/
# - data/
# - feature_files/
# - Additional CSV or JSON files as needed

# 8. Run the web application
cd app
python app.py

# Access the local server at: http://127.0.0.1:5000/
# 1. Clone the PrediLnc GitHub repository to your local system
git clone https://github.com/Udit64/PrediLnc.git
cd PrediLnc

# 2. Install Anaconda or Miniconda (choose one depending on preference)
#    These provide a Python/R environment manager. This step is manual:
#    - Anaconda: https://www.anaconda.com/products/distribution
#    - Miniconda: https://docs.conda.io/en/latest/miniconda.html

# 3. Create a new Conda environment named 'rtest' with Python 3.10 and R 4.1.2
#    It also installs rpy2 (Python-R interface) and r-essentials (base R packages)
conda create -n rtest python=3.10 r-base=4.1.2 rpy2 r-essentials -c conda-forge

# 4. Activate the environment
conda activate rtest

# 5. Install all required Python dependencies using pip and requirements.txt
#    This includes Flask, scikit-learn, numpy, pandas, and more
pip install -r requirements.txt

# 6. Install Redis 7.2.1 from source (required for real-time communication)
git clone https://github.com/redis/redis.git
cd redis

# Switch to the specific Redis version tag
git checkout 7.2.1

# Compile the Redis source code using all available cores
make -j

# Start the Redis server in the background
src/redis-server &

# 7. Confirm Redis is running by checking the server response
#    You should see the output: PONG
redis-cli ping

# Return to the PrediLnc root directory
cd ..

# 8. Download the pretrained model, processed feature matrices, and data files from Zenodo
#    You need a valid access token to use zenodo_get
zenodo_get --access-token G46P8DtW8lfKSG0u7IUVmCMb4idEKAoDCBL1yHWuwUkKvnFuGPSNCIkCham2 15764921

# 9. Unzip the downloaded dataset archive into the app/ directory
unzip saved_dataset.zip -d app/

# Make sure the app/ directory now contains:
# - models/          → trained machine learning models
# - data/            → processed datasets and metadata
# - feature_files/   → sequence, ontology, and interaction features
# - CSV/JSON files   → supporting input/output mappings

# 10. Move into the app/ directory to launch the Flask web application
cd app

# 11. Start the local Flask server. This will launch the web interface.
python app.py

# 12. Open your browser and visit the local web interface at:
#     http://127.0.0.1:5000/
```
# Workflow
![Workflow_final 1](https://github.com/user-attachments/assets/aab4ff5d-640d-4396-9203-43b33a50358d)
