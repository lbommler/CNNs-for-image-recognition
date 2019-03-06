from PIL import Image
import numpy as np
import pandas as pd
import os 
import timeit


# 1) Import as pd.DataFrame
WDPath = "C:/Users/LukasPC/OneDrive/Studium/Master/5 - WS 18 19/Machine Learning Seminar/Data"
os.chdir(WDPath) 
train = pd.read_csv("train_bitmaps.csv", index_col=0, header = None)
train.columns = ["result"] # rename column name
print(train.shape)
print(train.head())
target = np.array([int(item) for picture in train["result"] 
                   for item in picture]) # get the results of all 576000 blocks
print(target.shape)

# Funktion zur Iteration über quadratische Bildausschnitte mit Seitenlänge block_size
def split_image(img, block_size):
    for i in range(img.size[1]//block_size):
        for j in range(img.size[0]//block_size):
            area = [j*block_size, i*block_size, (j+1)*block_size, (i+1)*block_size]
            yield img.crop(area).resize((30,30), Image.ANTIALIAS)
        

piclist = []
blocks = []
for picNr in range(1000,2000):
    start = timeit.default_timer()
    # Import des Bilds  
    img = Image.open("TrainingImages/"+str(picNr)+".jpg")#.convert("L")
    for block in split_image(img, 120):
        #print(list(block.getdata()))
        blockData = block.convert("L")
        blockData_flat = np.asarray(blockData).flatten('F')
        #print(blockData)
        #print(blockData_flat)
        blocks.append(blockData_flat.tolist())       
    #piclist.append(blocks)
    print("Picture "+str(picNr)+" processed: " + str(round(timeit.default_timer() - start,2)) )
    
    
piclist = np.float32(np.asarray(blocks[:]))
data = piclist
#df_blocks.to_pickle("data.pkl")

#from sklearn import svm

#model = svm.SVC()

#X = df_blocks.as_matrix
#y = target

#model.fit(X,y)
