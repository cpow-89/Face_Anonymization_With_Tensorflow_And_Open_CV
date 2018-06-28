# Face_Anonymization_With_Tensorflow_And_Open_CV

### What does the project?

- face recognition with open CV
  - works on images
- facial key point detection 
  - using the output of the open CV face detector as input for a CNN model built in Tensorflow with outputs the facial key points
  - works on images
- face anonymization 
  - uses the open CV face detector -> facial key point detection pipeline 
  - adds an overlay image to the detected facial key points 
  - works on images

### Requirements?
- I used an anaconda installation with the following additional packages:

python - 3.6.5 <br>
opencv3 - 3.1.0 <br>
tensorflow-gpu - 1.8.0 <br>
pandas - 0.23.0 <br>
scikit-learn - 0.19.1 <br>  
numpy - 1.14.2 <br>
matplotlib - 2.2.2 <br>

### Dataset Source:
https://www.kaggle.com/c/facial-keypoints-detection/data
