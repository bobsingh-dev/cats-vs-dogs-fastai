
---

## 🚀 How It Works
1. **Download dataset** (Oxford-IIIT Pets, 37 cat/dog breeds).  
2. **Prepare data** using a fastai `DataBlock`:  
   - Images as inputs.  
   - Labels extracted from filenames (e.g., `Siamese_34.jpg` → `Siamese`).  
   - 80/20 train-validation split.  
   - Images resized to 224x224 pixels.  
3. **Train model** using transfer learning with a pretrained **ResNet18** CNN.  
4. **Evaluate model** on validation set using error rate.  
5. **Predict** the breed of a new image.  

---

## 📊 Example Output
```text
Prediction: Siamese
Confidence: 0.9987
