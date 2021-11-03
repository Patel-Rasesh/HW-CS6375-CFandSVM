from posixpath import normcase
from numpy.core import numeric
#from numpy.core.numeric import NaN
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
import timeit
import math
'''
functions :
findSimilarUsers -  finds similar users (user list) to the user passed as an argument
predictRating    -  predicts ratings and stores them in an array
evaluation       -  computes Mean Absolute Error and RMSE
'''
#pathToTrain = "C:/Users/Friday/Desktop/Fall21/CS6375/Homework2/TrainingRatings.txt"
pathToTrain = "/TrainingRatings.txt"
pathToTest = "/TestingRatings.txt"

start = timeit.default_timer()
#Reading training and testing file in a respective data frame
print("Enter path to training file and test file respectively when asked. Or press Enter to use the same files which were given in the assignment.")

print("Enter path to training file : \n")
pathToTrainingFile = input()
if(pathToTrainingFile == "\n"):
    pathToTrainingFile = pathToTrain
dfInput = pd.read_csv("C:/Users/Friday/Desktop/Fall21/CS6375/Homework2/TrainingRatings.txt")
dfInput.columns = ['Movies', 'Users', 'RawRatings']

print("Enter path to test file : \n")
pathToTestFile = input()
if(pathToTestFile == "\n"):
    pathToTestFile = pathToTest
dfOutput = pd.read_csv("C:/Users/Friday/Desktop/Fall21/CS6375/Homework2/TestingRatings.txt")
dfOutput.columns = ['Movies', 'Users', 'TestRatings']

print("Computing vote matrix and similarity between users ...\n")
print("Please wait. Prediction time for entire test file is around 18-20 mins ...")

#Computing mean vote and merging it to the original training data frame
dfMeanVote = dfInput.copy().drop(columns='Movies')
dfMeanVote = dfMeanVote.groupby(['Users'], as_index=False, sort=False).mean().rename(columns = {'RawRatings':'MeanRatings'})
dfInput = pd.merge(dfInput, dfMeanVote, on='Users', how='left', sort=False)

#Forming a matrix (dimensions - users x movies) with raw ratings
dfInputMatrixRawRatings = pd.DataFrame({"Users":dfInput['Users'],
                            "Movies":dfInput['Movies'],
                            "Ratings":dfInput["RawRatings"]})
dfInputMatrixRawRatings = dfInputMatrixRawRatings.pivot_table(index='Users',columns='Movies',values='Ratings').fillna(0)

# Computing correlation similarity (Faster computation)
cosineSimilarity = cosine_similarity(dfInputMatrixRawRatings)
dfWeightMatrix = pd.DataFrame(cosineSimilarity)

#weightMatrixNumpy = np.corrcoef(dfInputMatrixRawRatings)
#dfWeightMatrix = pd.DataFrame(weightMatrixNumpy)
listOfUsers = np.asarray(dfInputMatrixRawRatings.index)
dfWeightMatrix.columns = listOfUsers
dfWeightMatrix.index = listOfUsers

def findSimilarUsers(user):
    '''
    This gathers a list of users who are similar to user passed as parameter
    Remove users who have not voted atleast 50 movies
    '''
    similarUsers = np.array([])
    ratedMovies = np.array([])
    
    for i, value in enumerate(np.asarray(dfWeightMatrix[user])):
        #Only consider a user if similarity (between activeuser and user) is higher than X
        if value > 0.6:
            neighborMovies = np.asarray(dfInputMatrixRawRatings.loc[listOfUsers[i],:])
            ratedMovies = np.count_nonzero(neighborMovies)
            #User (to be considered a neighbor of active user) needs to have voted atleast X movies
            if ratedMovies > 10:
                #Only first X users are considered good
                if(similarUsers.shape[0] <= 10):
                    similarUsers = np.append(similarUsers, listOfUsers[i])
                else:
                    break
    return similarUsers

def predictRating(dfOutput):
    '''
    Predicts ratings for the combination (of user and movie) given in the test file
    '''
    predictedValues = np.array([])
    for i in range(len(dfOutput)):
        print("Entered in prediction line", i)
        normConstant = 0
        keppaConstant = 0
        collectedWeight = 0
        similarUsers = findSimilarUsers(dfOutput['Users'][i])
        for user in similarUsers:
            if dfInputMatrixRawRatings.loc[user][dfOutput['Movies'][i]] != 0.0:
                collectedWeight += dfWeightMatrix.loc[dfOutput['Users'][i], user]*(dfInputMatrixRawRatings.loc[user][dfOutput['Movies'][i]] - dfInput['MeanRatings'].loc[dfOutput['Users'][i]])
                normConstant += np.abs(dfWeightMatrix.loc[dfOutput['Users'][i], user])
                keppaConstant = 1/normConstant
        predictedValues = np.round(np.append(predictedValues, (keppaConstant*collectedWeight) + dfInput['MeanRatings'].loc[dfOutput['Users'][i]]), 1)
        print("Exiting from prediction ", i)
    return predictedValues

def evaluation(predictedValues, dfOutput):
    '''
    Calculates how far predicted values are from true values
    '''
    trueValues = np.asarray(dfOutput['TestRatings'])

    meanAbsError = mean_absolute_error(trueValues, predictedValues)
    RME = np.square(np.subtract(predictedValues,trueValues)).mean()
    RMSE = np.sqrt(RME)
    return meanAbsError, RMSE

#PredictedValues are the predicted ratings
predictedValues = predictRating(dfOutput)

meanAbsError, RMSE = evaluation(predictedValues, dfOutput)
print("Prediction evaluation metric : Mean Absolute Error = ", meanAbsError)
print("\t\tRoot Mean Square Error = ", RMSE)

stop = timeit.default_timer()
print("Runtime = ", stop-start)