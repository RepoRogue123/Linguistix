# 🎙️ Linguistix: Speaker Recognition System

**CSL2050 - Pattern Recognition and Machine Learning Project**  
_Indian Institute of Technology, Jodhpur_

## 👥 Team Members
- Shashank Parchure (B23CM1059)  
- Atharva Honparkhe (B23EE1006)  
- Vyankatesh Deshpande (B23CS1079)  
- Abhinash Roy (B23CS1003)  
- Namya Dhingra (B23CS1040)  
- Damarasingu Akshaya Sree (B23EE1085)

## 📌 Abstract

**Linguistix** is a speaker recognition system that applies classical ML techniques to identify speakers from voice samples. It explores a wide range of models including KNN, SVM, Decision Trees, ANN, Naïve Bayes, GMMs, and K-Means. Dimensionality reduction (PCA, LDA) and ensemble learning methods (Bagging, Boosting, Stacking) are integrated for performance optimization.

> 🧠 **Key Insight**: Supervised dimensionality reduction (especially LDA) significantly boosts the accuracy and generalization of traditional ML models in speaker recognition.

---

## 📂 Dataset

- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/vjcalling/speaker-recognition-audio-dataset)
- **Samples**: 2511 `.wav` files
- **Speakers**: 50 unique identities
- **Feature Extraction**: MFCCs → 4000-dimensional vectors per sample

---

## 🛠️ Tech Stack

- **Language**: Python 3.10
- **Libraries**: `librosa`, `numpy`, `scikit-learn`, `pandas`, `PyTorch`, `matplotlib`
- **Environment**: Jupyter Notebooks
- **Cloud Deployment**: Google Cloud VM (4GB RAM, 25GB Disk)

---

## 🧪 Models Implemented

### 🔹 Supervised Classifiers
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machines (SVM)**
- **Decision Trees**
- **Naïve Bayes**
- **Multi-Layer Perceptrons (MLP)**
- **Artificial Neural Networks (ANN)**

### 🔹 Clustering Techniques
- **K-Means**
- **Gaussian Mixture Models (GMM)**

### 🔹 Ensembles
- **Bagging**
- **AdaBoost & Improved AdaBoost (SAMME)**
- **Stacking (SVM + GMM + RF)**

### 🔹 Dimensionality Reduction
- **PCA**
- **LDA**
- **t-SNE & UMAP (for visualization)**

---

## 📊 Key Results (Test Accuracy)

| Model                          | Test Accuracy (%) |
|-------------------------------|--------------|
| **ANN with LDA**              | **100.00**   |
| **KNN with LDA**              | 99.80        |
| **Bayesian with LDA**         | 99.80        |
| **SVM with PCA**              | 99.40        |
| **GMM (Semi-supervised)**     | 99.67        |
| **CNN with LDA + PCA**        | 99.80        |
| **Decision Tree + Bagging**   | 82.70        |
| **K-Means with LDA + PCA**    | 87.67        |

---

## 📌 Key Observations

- **LDA > PCA**: LDA consistently performed better across models due to class-separability.
- **GMM + Supervision**: Semi-supervised GMMs delivered the best clustering results.
- **ANN + LDA**: Delivered perfect classification across all splits.
- **Ensemble Methods**: Bagging and AdaBoost helped reduce overfitting in tree-based models.

---

## 🌐 Live Demo & Resources

- 🔗 [**GitHub Repository**](https://github.com/RepoRogue123/Linguistix)  
- 🎥 [**Spotlight Video**](https://youtu.be/yORB3cY9WDA)  
- 💻 [**Web Demo** (Hosted on Google Cloud)](http://34.121.3.96:8080/)  
- 📄 [**Project Page**](https://vyankateshd206.github.io/Linguistix/)  

---

## 🤝 Contributions

### **Shashank Parchure (B23CM1059)**
- **Implemented**:  
  - KNN with PCA  
  - Bayesian Learning with Correlation-based Feature Selection  
  - SVM with PCA  
  - MLP with PCA  
  - ANN with LDA  
- **Responsible for**:  
  - Report compilation  
  - Spotlight video organization (content outline)  
  - Deploying demo code on Google Cloud  

### **Atharva Honparkhe (B23EE1006)**
- **Implemented**:  
  - Decision Tree with PCA, LDA, and Ensemble Methods  
  - GMM  
- **Responsible for**:  
  - Demo code creation  
  - Report compilation  

### **Vyankatesh Deshpande (B23CS1079)**
- **Implemented**:  
  - KNN with LDA  
  - Bayesian Learning with LDA  
  - SVM with Correlation-based Feature Selection  
  - ANN with PCA  
- **Responsible for**:  
  - MFCC feature extraction  
  - Project page implementation  
  - Report compilation  
  - Google Cloud exploration  

### **Abhinash Roy (B23CS1003)**
- **Implemented**:  
  - KMeans Clustering with LDA  
  - Decision Tree on raw data  
  - CNN with PCA and LDA  
- **Responsible for**:  
  - Spotlight video creation  

### **Namya Dhingra (B23CS1040)**
- **Implemented**:  
  - Decision Tree with UMAP and t-SNE  
  - KMeans Clustering with PCA  
- **Responsible for**:  
  - Spotlight video presentation organization  
  - Content writing  

### **Damarasingu Akshaya Sree (B23EE1085)**
- **Implemented**:  
  - KNN  
  - Bayesian Learning with PCA  
  - SVM with LDA  
  - MLP with Correlation-based Feature Selection  
  - ANN with Correlation-based Feature Selection  
- **Responsible for**:  
  - Spotlight video presentation organization  
  - Content writing  

---

## 📚 References

- [NumPy Documentation](https://numpy.org/doc/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Understanding Bootstrapping – Medium](https://medium.com/@wl8380/understanding-the-bootstrapping-process-in-machine-learning-a6372bf7b4e2)
- [Ensemble Methods – Medium](https://medium.com/@shashank25.it/ensemble-methods-in-machine-learning-2d4cc7513c77)
