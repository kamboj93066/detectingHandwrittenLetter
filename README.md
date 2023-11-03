# detectingHandwrittenLetter
This repo contains the files used to make a handwritten recognition system from start to end.
Here we have three .ipynb files.
1. First one with the name as conversion.ipynb, this file has the code to convert the csv dataset into images. This code reads the csv file row by row and convert the given pixel values to images. Images are saved in separate folder. In that folder different sub-folders are creates such as: sub-folder A contains the images of letter A, sub-folder B contains the images of letter B and so on.
2. The second ipynb file is recognition.ipynb, this file contains the python code to train our CNN model. This code works by reading the files containing images and separating the data into training and testing data. This also preprocess them and then a CNN model is created and then the data is fed to the model. The model is then trained and then is tested on the test data and the accuracy is found. At last the model is saved which can be seen in the directory named model.
3. The third ipynb file is using_model.ipynb, this file contains the python code to run the model which was saved in the previous file. This code takes the image path and predicts the model output.
4. The fourth directory is named as WebApp, this directory contains the app.py file which contains the flask code that is used to host a small webapp on local machine. This file contains the predict function that takes the image path which is taken by the flask code.
5. In the same directory we have another directory named templates, in this directory we have a html file that contains the code to create a simple webpage that is used by the flask to take image from the user and show the output.
