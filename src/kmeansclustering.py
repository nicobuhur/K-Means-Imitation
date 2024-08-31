"""
Created on Sun Jun 22 23:36:37 2021

@author: Necati Buhur
"""

from math import sqrt
import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd
from PCA import PCA
np.random.seed(1234)

cols_to_use = ['fixed acidity', 'volatile acidity', 'citric acid', \
               'residual sugar', 'chlorides',\
                'free sulfur dioxide', 'total sulfur dioxide',\
                    'density', 'pH', 'sulphates' , 'alcohol']
class KMeans:    
    def __init__(self,k, max_iters ,plot_steps):#this function is essential for starting the code
        self.k = k                              #since important values are added in the test code thanks to
        self.max_iters = max_iters              #this function.
        self.plot_steps = plot_steps                
    def predict(self, data1):# this function is the most important function since it does the cluster prediction
        pca = PCA(2) 
        pca.fit(data1)
        data1 = pca.transform(data1)#making the data 2D        
        old_centroids=pd.DataFrame(np.random.rand(6,11), columns= cols_to_use)#These are the very first random centroids
        old_centroids=pca.transform(old_centroids)
        centroids = pd.DataFrame(np.random.rand(6,11), columns= cols_to_use)#Also, These are the same, but I need two of them 
        centroids=pca.transform(centroids)        
        centroids = pd.DataFrame(np.random.randn(6,2), columns=['x','y'])#I made dataframe since it is easy to handle
        old_centroids = pd.DataFrame(np.random.randn(6,2), columns=['x','y'])
        data1= pd.DataFrame(data1, columns=['X','Y'])#dataframe for ease the process and understanding the data
        deneme=0
        while deneme < self.max_iters: #this while loop will continue untill deneme number is over 200.
            point_locations= pd.DataFrame(columns=['cluster','X','Y'])             
            my_cluster_prediction=[]# this list will contain cluster information for each point in our data.                        
            for i in data1.index:#for loop searches all the data points 
                centroiddistances=[]#this list will contain centroid distances for each point, but in each loop, its information will change.    
                for k in range(self.k):#self.k means the cluster number
                    dist= np.linalg.norm(data1.iloc[[i]].to_numpy() - centroids.iloc[[k]].to_numpy()) #find distance between one point and each centroid.                    
                    centroiddistances.append(dist)
                my_kluster_info=centroiddistances.index(min(centroiddistances)) #find the smallest distance, then take the information of the centroid, which is the closest                                
                point_locations.loc[i,'X']= (data1.loc[i,'X'])
                point_locations.loc[i,'Y']= (data1.loc[i,'Y'])
                point_locations.loc[i,'cluster']=my_kluster_info
                my_cluster_prediction.append(my_kluster_info)                        
                point_locations=point_locations.sort_values(by=['cluster'])            
            def closest(lst, K):# this function will find the closest value to intended value.      
                return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]            
            group_colors = ['black', 'gray', 'orange', 'pink','purple','cyan']
            group_names= ['cluster 0','cluster1','cluster2','cluster3','cluster4','cluster5']
            for j in np.unique(point_locations['cluster']):            
                meanx=0
                meany=0
                numberx=[]
                numbery=[]                
                for i in data1.index:                    
                    if point_locations['cluster'][i]==j:                        
                        numberx.append(point_locations['X'][i])
                        numbery.append(point_locations['Y'][i])
                        meanx+=point_locations['X'][i]
                        meany+=point_locations['Y'][i]
                        colors = [group_colors[j]]
                        labels = [group_names[j]]
                        plt.scatter(point_locations['X'][i], point_locations['Y'][i],c=colors)                                                       
                meanx=meanx/len(numberx)
                meany=meany/len(numbery)                
                meanx=closest(numberx, meanx)
                meany=closest(numbery, meany)                
                centroids.loc[j,'x']=meanx
                centroids.loc[j,'y']=meany
                plt.scatter(centroids['x'][j],centroids['y'][j], marker='x', c='red')            
            plt.title('Wine Dataset')    
            #plt.legend(loc='best')
            plt.show()
            deneme+=1
            print('Iteration number',deneme)            
            if deneme==20:            
                break
        return my_cluster_prediction #this return belongs to predict function. This information is needed for comparison.
                                     #The comparison is either making clusters using 'quality' column or making prediction using whole data
                        
# ===========================================================================================================================================
# A proof that np.linalg.norm function and euclidian distance formula gives equal products                   
# ===========================================================================================================================================
# centroids = pd.DataFrame(np.random.randn(6,2), columns=['x','y'])
# data1 = pd.DataFrame(np.random.rand(18,2),columns=['X','Y'])
# numbers=[]
# for i in range(18):
#     numbers.append(random.randrange(0,6))

# for i in data1.index:
#     distances=[]
#     distances1=[]
#     for k in range(6):   
#         dist= np.linalg.norm(data1.iloc[[i]].to_numpy() - centroids.iloc[[k]].to_numpy())
#         dist1=sqrt((data1.loc[i,'X']-centroids.loc[k,'x'])**2 + (data1.loc[i,'Y']-centroids.loc[k,'y'])**2)
                   
#         distances.append(dist)
#         distances1.append(dist1)
# print(distances==distances1)                    
  
