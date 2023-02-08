# COMPARATIVE VISUALIZATION OF DIMENSION REDUCTION TECHNIQUES

### USER MANUAL

A file ‘dim_reduce.py’ contains all the algorithms and the code for the application. 
The user needs to start the command line and type ‘python dim_reduce.py’ to start the application.
A pop up appears on the screen to upload the file. This file can be selected from any of the folders on the local machine. 
The data file needs to be a comma separated with the target column as the last column of the dataset. 
The algorithms implemented are Isomap, TSNE and PCA. All the three algorithms work only on continuous independent variables with multi-categorical target variables.

The following parameters are necessary for the application to work –
•	Isomap 

    o	Nearest Neighbors(default = 5)
    
    o	No of dimensions
    
•	TSNE 

    o	Nearest Neighbors(default = 30)
    
    o	No of dimensions
    
    o	No of iterations(default = 1000)
    
•	PCA 

    o	No of dimensions
    
    
The no of dimensions must be either 2 or 3.
Note:- The number of dimensions for all the 3 algorithms must be the same. If the dimension is different, then the application will throw an error.




