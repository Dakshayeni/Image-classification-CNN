# Image-classification-CNN

# 🐶🐱 Cat vs Dog Image Classification  

## 📌 Project Overview  
This project aims to build an image classification model to distinguish between images of cats and dogs. The dataset is sourced from Kaggle's *Dogs vs. Cats* competition, containing 25,000 labeled images for training. Various CNN architectures were tested to achieve optimal accuracy while minimizing overfitting.  

## 📂 Dataset  
- **Source:** [Kaggle - Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats)  
- **Training Data:** 25,000 labeled images (cats & dogs)  
- **Test Data:** Unlabeled images for prediction  
- **Labels:** 1 (Dog), 0 (Cat)  

## 🏗️ Project Workflow  
1. **Data Preprocessing**  
   - Extract and organize images  
   - Assign labels based on filenames  
   - Split dataset (80% train, 20% validation)  

2. **Model Selection**  
   - Convolutional Neural Networks (CNN) chosen over DNN for spatial feature extraction  
   - Five different CNN architectures were tested  

3. **Training & Optimization**  
   - Used data augmentation & dropout to prevent overfitting  
   - Compared activation functions (ReLU vs. LeakyReLU)  
   - Applied batch normalization & L2 regularization  
   - Trained models for 20 epochs to achieve high accuracy  

4. **Evaluation & Results**  
   - Trained five different CNN models  
   - Model 4 performed best (final validation accuracy: **84.12%**)  
   - ResNet was considered but required higher computational power  

5. **Predictions & Submission**  
   - Predicted results stored in `submission.csv`  
   - Used trained model to classify new images  

## 🧠 Model Architectures  
### ✅ **Best Model (Model 4)**
- **Layers:**  
  - 4 Convolutional layers  
  - Max-pooling after each conv layer  
  - Dense layer with 512 neurons + dropout (50%)  
- **Regularization:** Dropout (25%), L2 Regularization  
- **Optimizer:** Adam  
- **Loss Function:** Binary Crossentropy  
- **Final Accuracy:** **94.12%**  

### 📊 **Comparison of Models**  
| Model | Accuracy | Overfitting | Regularization | Dropout | Activation |
|--------|---------|-------------|---------------|---------|------------|
| Model 1 | 86% | Minimal | No | No | ReLU |
| Model 2 | 87% | Controlled | No | No | LeakyReLU |
| Model 3 | 86.56% | Minimal | Yes (L2) | Yes (50%) | ReLU |
| **Model 4 (Best)** | **94.12%** | **Well-controlled** | **Yes** | **Yes** | **ReLU** |
| Model 5 | 56% | High | Yes | Yes | ReLU |

## 📌 Key Takeaways  
✅ CNNs are better suited for image classification than fully connected DNNs  
✅ Dropout & L2 Regularization significantly reduced overfitting  
✅ Model 4 achieved the best balance between accuracy and generalization  
✅ Further improvements can be made with hyperparameter tuning and deeper architectures  

## 🚀 Future Improvements  
- Increase training epochs (50-100) for better performance  
- Use advanced architectures like ResNet or EfficientNet  
- Optimize model hyperparameters for improved accuracy  

## ⚡ Running the Code  
### 1️⃣ Install Dependencies  
```bash
pip install tensorflow keras numpy pandas matplotlib
```
### 2️⃣ Train the Model  
Run the Jupyter Notebook:  
```bash
jupyter notebook EE628_Final_Project.ipynb
```
### 3️⃣ Predict & Generate Submission  
Execute the model prediction and save results in `submission.csv`.

