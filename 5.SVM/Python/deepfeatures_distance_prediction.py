# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 08:58:43 2015

@author: hehu
"""

# Example of prediction using deep features.
# heikki.huttunen@tut.fi; 28.10.2015

import os
import random
import glob
import numpy as np

if __name__ == "__main__":
    
    trainFolder = "train"
    testFolder = "test"
    pairsFile = "pairs.csv"

    # Load all deep features for test files into memory

    files = glob.glob(testFolder + os.sep + "*_deepfeatures.csv")
    features = {}
   
    index = 0
        
    for name in files:    
        feat = np.loadtxt(name)
        basename = os.path.basename(name).split("_")[0]

        features[basename] = feat
        
        if index % 100 == 0:
            print "%d/%d files loaded..." % (index, len(files))
        
        index += 1
        
    print "All files loaded. Computing distances for pairs."
	
    # Compute Euclidean distances for all requested image pairs
        
    f = open("pairs.csv", "r")

    # Store all distances here:

    distances = []
    index = 0
    
    # Read the file line by line:
    
    for line in f:
        
        if line.startswith("Id"): # This is the first line; skip it
            continue
        
        # Read the line. Strip removes whitespaces on both sides 
        # and split splits the line by the delimiter ",".
        
        Id, file1, file2 = line.strip().split(",")
        
        # Find the deep features related to the two files:
        
        feat1 = features[file1]
        feat2 = features[file2]
    
        # Compute the distance:
    
        distance = np.linalg.norm(feat1 - feat2)
        distances.append(distance)        
        
        if index % 1000 == 0:
            print "%d/901152 pairs done..." % (index)
        
        index += 1
        
    f.close()

    # Find largest distance
    maxDistance = np.max(distances)
   
    # Create submission file

    f = open("submission.csv", "w")    
    f.write("Id,Prediction\n")
    
    for index in range(len(distances)):
        
        similarityScore = (maxDistance - distances[index]) / maxDistance
        f.write("%d,%.6f\n" % (index, similarityScore))
        
    f.close()
	
    print "Prediction finished."
	