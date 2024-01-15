# Siamese_Duplicates
In this project, we proposed a Siamese Network to detect near-duplicates of scanned documents. We specifically focused on documents such as forms, invoices and receipts. Furthermore, we implemented popular image classification architectures, AlexNet, ResNet-18 and VGG-16 into our Siamese model and also introduced different loss functions such as Triplet Loss and Contrastive Loss. Feel free to reach out to me for the full research paper. 

## 1. Data 
The forms data were sourced from the FUNSD dataset (Guillaume et al., 2019), while the invoice sdata were retrieved from Mendeley Data (Kozlowski et al.,
2021). Additionally, the receipts data were collected from the ICDAR-SROIE dataset (Zheng et al., 2019).

## 2. Data Augmentation 
We introduced five different type of image transformation
techniques:
- 1. Rotation and translation - Simulating misalignments and displacements in scanned documents within a max range of five degrees.
- 2. Random noise - Adding Gaussian and Speckle
noise to mimic scanning irregularities.
- 3. Random zoom - Randomly create a 90% to 110% scaled image.
- 4. Random occlusions - Blocking parts of images to simulate obscured or damaged sections.
- 5. Random copy-move edits - Using OCR to copy and paste random text, introducing editing scenarios.
 
## 3. Experiment Results(Best model) 
 Model  | Loss Function | Accuracy | F1 Score |   
| ------------- | ------------- | ------------- | ------------- | 
| Vanila | Triplet Loss | 0.856 | 0.787 | 
| ResNet | Triplet Loss | 0.869 | 0.806 | 
