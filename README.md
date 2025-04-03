SAM (Segment Anything Model) Implementation for One-Shot Detection.
This project is an implementation of 'Personalize Segment Anything Model with One Shot'(https://arxiv.org/pdf/2305.03048).
This repository provides an implementation of the Segment Anything Model (SAM) for one-shot detection using Vision Transformers (ViT). The model leverages cross-attention layers and target-semantic prompting to perform accurate segmentation of objects in a given image. It utilizes a reference image and its mask for target-guided attention, and a test image is processed with the model to predict the segmentation mask.

The key features of the code include:

Preprocessing images and masks.

Feature extraction using the SAM model.

Cosine similarity-based localization of objects in the test image.

Peak local maxima detection to find key points for segmentation.

Cascaded post-refinement stages for better mask prediction.


Requirements
Python 3.7+

PyTorch (1.10.0+)

transformers library (from Hugging Face)

torchvision

matplotlib

scikit-image

OpenCV

Install dependencies using pip:
pip install torch torchvision transformers matplotlib scikit-image opencv-python



Project Overview
The code implements a process for detecting objects using a reference image and a test image. The steps are as follows:

Preprocessing:

The reference image and its corresponding mask are preprocessed to match the target size and format for model input.

The test image is also preprocessed and prepared for similarity calculations.

Feature Extraction:

The model extracts embeddings from both the reference and test images.

Cosine similarity is calculated between the reference and test image embeddings to identify possible object locations.

Point Selection:

Top-k positive and negative points are selected based on the similarity map of the test image to guide the attention mechanism.

Model Inference:

The model uses the selected points and attention maps to perform one-shot segmentation and refine the predicted mask.

Cascaded Post-Refinement:

The first post-refinement step improves the mask predictions, and a second step further refines the mask using the best IoU (Intersection over Union) score.

Mask Visualization:

The final segmentation mask is overlaid on the test image for visualization.

Code Walkthrough
1. Preprocessing Functions
get_preprocess_shape(): Computes the new dimensions for resizing the input image to a target size.

preprocess(): Normalizes pixel values and pads the input image to a square shape.

prepare_mask(): Prepares the reference mask for processing, including resizing and normalizing.

2. Image and Mask Encoding
The AutoProcessor and SamModel from Hugging Face’s transformers library are used to load the SAM model and processor.

The reference and test images are passed through the model to extract embeddings.

3. Cosine Similarity for Localization
Cosine similarity is calculated between the reference and test image embeddings to find the most similar regions in the test image.

A similarity map is generated, and the top-k peaks are identified as potential object locations.

4. Point Selection and Prediction
Points with the highest similarity are selected to guide the attention mechanism.

The model is then used for segmentation, and the masks are refined through cascaded post-processing.

5. Visualization
The predicted mask is visualized by overlaying it on the test image, with detected peaks marked.
