
#LBPH-vs-LMDeP
it is a comparitive study between two face feature extraction method(LBPH and LMDeP) using various classifiers.

#Dependencies
*CV2
*scikit Learn
*Pandas
*sklearn
*python 3.7
*pyQT5
*numpy

#To Run
to start the program just run main2.py program.

! WARNING !under Progress
* Currently the recogniser does not work well in dynamic environment,So i am working towards integrating a deep learning model.
* LMDeP option in standard tab, Takes a lot of time to process 200 images,near about 25 minutes.As the LMDeP was applied from scratch thus it is not optimised,i am working towards the same

#How to use

'ADD'
Add tab wants the user to enter his/her face data for feature extraction,

1.Enter name then press 'ENTER'
2.then press the capture button(you have to bring your face bit closer to web cam to start recognition process) 
3.then capture 20 images per person and press process button for starting the face feature extraction method using LBPH.
4.then press next button to add other faces otherwise

'RECOGNISE'
5.move to recognition tab and press start.
6.bring face closer, and press recognise button to start recognition.
7.Name of the person will be visible in the right side of the program window as entered earlier.

'STANDARD'
here you can check the precision,recal,accuracy of face feature extraction method using various classifier to do a comparision.

1.Select the data set,the dataset are standard and are in growing complexities,you can check the same in the folder.
2.next select the no of data you want to process, max is 2000.
3.next select the feature extraction method.
4.select the classifier.
5.select the % of training data you want to provide.
6.press the process button.

'ANALYSIS1'
here you can compare the acuuracy of two feature extractor on the basis of classifier.the graph varies slightly everytime as it is a classification problem.

1.Press LBPH button wait for about 20 sec the result will be viewed(toggle to full screen for better readability)
2.Press LMDeP button wait for about 20 sec the result will be viewed.


