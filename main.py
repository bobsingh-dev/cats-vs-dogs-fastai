# -----------------------------------------------------------
# Simple Image Classification with fastai (Cats vs Dogs)
# Dataset: Oxford-IIIT Pets (37 breeds of cats and dogs)
# Model: ResNet18 (transfer learning)
# -----------------------------------------------------------

# 1. Import the fastai vision library
from fastai.vision.all import *

# 2. Download and point to the dataset
# - untar_data(URLs.PETS) downloads the Oxford-IIIT Pets dataset
# - The images are inside the "images" subfolder
path = untar_data(URLs.PETS) / "images"

# 3. Define a simple label function
# Example: "Siamese_34.jpg" -> "Siamese"
def label_from_fname(fname):
    return fname.name.split('_')[0]

# 4. Create a DataBlock (blueprint for dataset)
# - ImageBlock: input images
# - CategoryBlock: output labels (cat/dog breeds)
# - get_items: find image files in the dataset
# - splitter: split data into 80% training, 20% validation
# - get_y: how to extract labels from filenames
# - item_tfms: resize all images to 224x224 pixels
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=label_from_fname,
    item_tfms=Resize(224)
).dataloaders(path, bs=32)  # build DataLoaders with batch size of 32

# 5. Show a few sample images with their labels
dls.show_batch(max_n=6)

# 6. Create and train the model
# - vision_learner: builds a CNN with transfer learning
# - resnet18: a small but powerful pretrained CNN backbone
# - metrics=error_rate: tracks model accuracy
learn = vision_learner(dls, resnet18, metrics=error_rate)

# Fine-tune the model:
# - Step 1: train only the new head (frozen backbone)
# - Step 2: unfreeze backbone and train entire model
learn.fine_tune(1)

# 7. Test the model on a single image
img = PILImage.create(path/'Siamese_34.jpg')
pred_label, _, probs = learn.predict(img)

# Print results
print(f"Prediction: {pred_label}")
print(f"Confidence: {probs.max():.4f}")
