# LSFIA
Source code for the design of an ML model for real-time image recognition of sign language. 

## To construct a video dataset
Run the file recorddataset.py by defining a run configuration on Pycharm beforehand. For each sign, change the "collection" field with the right sign name to record. You can change the number of frames you want for each videos which will modify the length of the video. Please notice that the number of frames is the same for all the dataset you create otherwise you could get some troubles in the next step. 

## To process the videos and build the Numpy tables corresponding to each video
Prerequisites : have constructed a video dataset
Run the methods in main.py calling the functions in datacollection.py. 

## To build a new model and launch it after
Prerequisites : have processed a video dataset
Run the methods in main.py calling the functions of the file model.py and evaluation.py. Remember to set True as a parameter to the start_model() function in model.py. If you don't want to launch the new model don't run the method of the file evaluation.py.

## To launch a previously built model
Prerequisites : have built a model previously and have the dataset and the numpy arrays on the project you want to run the model from.
Put False in the parameter of the function start_model() of the file model.py. Also put the correct path to the .h5 file in the load_model() function of model.py. Then as above, run the methods in main.py calling the functions of the file model.py and evaluation.py. 

## Source : 

### Mediapipe
https://google.github.io/mediapipe/

### LSTM Deep Learning Model
https://www.youtube.com/watch?v=doDUihpj6ro
