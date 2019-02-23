# CMSC 733  Classical and Deep learning approaches to Geometric Computer Vision.
## Project 1: Auto-panorama

**Authors:**  
Darshan Shah  
Mayank Pathak  

To run the code, insert the Data folder with Train as subfolder in the Phase 1 directory. 
In the Phase 1/Code/Wrapper.py file, Modify the image path in Line 36 and 37 to point to image 1 and 2 respectively.

To run the program, open terminal and navigate to the Code directory and run the following command:  
```
python Wrapper.py
```  
The output generated by the program should be saved by the name "mypano.png" in the Code directory. In order to change the number of features detected, pass the argument in the following manner (Default = 100 features):  
```
python Wrapper.py --NumFeatures 60 
```  
the above command accepts 60 as number of features.  

***Phase 2: The Deep Learning**  

1. Ensure that the training data (MSCOCO) images are in the path as provided for this project.
2. Run Data_generation.py
3. Run Train.py   (No arguments are needed as the values are set to default). However, the values can be sent using arguments. More info in Train.py 
