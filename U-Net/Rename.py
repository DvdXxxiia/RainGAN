import os

path = "data/train_masks/"
files = os.listdir(path)
for index, file in enumerate(files):
    #print (file)
    os.rename(path+file, path + str(index+1)+ '_mask.png')