# 🩺 Skin Lesion Classification using Deep Learning  

Visit the website here: https://vishwaspuriofficial.github.io/SkinAI-Classifier/

<img width="1495" height="893" alt="image" src="https://github.com/user-attachments/assets/ae39ae24-ac57-4fcc-8a80-f644c1c3dca6" />


## 📌 Overview  
This project implements a deep learning pipeline to classify skin lesions into **7 categories** using the HAM10000 dataset.  
We compare a **baseline CNN** against a **transfer learning approach with ResNet18**, and apply **Grad-CAM** for interpretability.  
The project highlights the challenges of medical image classification and demonstrates the value of transfer learning in improving accuracy and efficiency.  

---

## ⚡ Key Features  
- 🔹 **Baseline CNN**: Custom-built convolutional neural network for benchmarking.  
- 🔹 **Transfer Learning (ResNet18)**: Fine-tuned pre-trained model achieving **79% test accuracy**.  
- 🔹 **Grad-CAM Heatmaps**: Visual explanations showing where the model “looks” when classifying lesions.  
- 🔹 **Evaluation on New Data**: Tested on unseen samples to confirm generalization.  
- 🔹 **Comparison of Architectures**: Analysis of efficiency and accuracy trade-offs.  

---

## 📊 Results  
- ✅ ResNet18 significantly outperformed the baseline CNN in accuracy and training efficiency.  
- ✅ Grad-CAM revealed that the model consistently focused on lesion regions, aligning with dermatologists’ diagnostic approach.  
- ⚠️ Most classes were well-identified, but **BCC (Basal Cell Carcinoma)** and **BKL (Benign Keratosis-like Lesions)** were harder to distinguish due to visual similarities.  
- 🎯 Overall test accuracy: **~79%**, which is considered strong for this task (70%+ is reasonable in literature).  

---

## 🧠 Takeaways  
- Skin lesion classification is inherently difficult due to visual similarity across classes.  
- Transfer learning provides substantial gains over training CNNs from scratch.  
- Grad-CAM improves trust and interpretability by validating that the model focuses on clinically relevant regions.  
- This project deepened our understanding of **model generalization, class imbalance, and the role of interpretability in medical AI**.  


