# Face-Verification
Face-Verification contest: https://inclass.kaggle.com/c/face-verification2

1.
  Face verification is a widely studied research topic, where the goal is to automatically determine whether two images represent the same person.
  The goal of this competition is to develop algorithms for accurate face verification. The goal is to learn to predict whether a pair of images represent the same person (target output = 1) or a different person (target output = 0).
  You can download the data(photos and their features) at https://inclass.kaggle.com/c/face-verification2/data.
  There are three files on the page.
  Pairs.zip contains many lines of photo ids in pair which we should calculte the possibility of each pair of photos to be the same person.
  Train.zip contains 1393 photos for 271 different people, each person's photos are in their own folder. We need to train a model by using the data in this file.
  Test.zip contains 1343 photos, these images are used for your verification submission (see pairs.csv).

2.
  I have already transformed the data in Train.zip,Pairs.zip and Test.zip into suitable format as training set and test set which could be loaded into Matlab, if you are using Matlab to program, please directly use the .mat files in the GitHub directory chenmengyang/Face-Verification/0.Data Set.
  
