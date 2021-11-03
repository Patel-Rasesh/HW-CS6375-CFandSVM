import os
import random
import timeit
import numpy as np
import itertools
import math
import pandas as pd
from collections import UserString, defaultdict
from sklearn.metrics import mean_squared_error

start = timeit.default_timer()
def extractDimensions(path):
    '''
    Returns the list of Movies and list of Users for the file kept at path
    '''
    listOfMovies = []
    listOfUsers = []
    trainingFile = open(path, 'r')
    for line in trainingFile:
        if(line != '\n'):
            #if(line.split(',')[0] not in listOfMovies):
            listOfMovies.append(int(line.split(',')[0]))
            listOfUsers.append(int(line.split(',')[1]))
    
    trainingFile.close()
    return listOfMovies, listOfUsers
    
inputListOfMovies, inputListOfUsers  = extractDimensions("C:/Users/Friday/Desktop/F21 Books/CS 6375/Homework 2/TrainingRatings_Sample.txt")
#outputListOfMovies, outputListOfUsers  = extractDimensions("C:/Users/Friday/Desktop/F21 Books/CS 6375/Homework 2/TestingRatings.txt")

#These returns 28978 and 1821
inputListOfUsersNumpy = np.unique(np.array(inputListOfUsers))
inputListOfMoviesNumpy = np.unique(np.array(inputListOfMovies))

#outputListOfUsersNumpy = np.unique(np.array(outputListOfUsers))
#outputListOfMoviesNumpy = np.unique(np.array(outputListOfMovies))

#Make a two dimensional array with (28978 * 1821, 2) dimensions
inputFeatureMatrix = np.array(np.meshgrid(inputListOfUsersNumpy, inputListOfMoviesNumpy))
inputFeatureMatrixNumpy = inputFeatureMatrix.T.reshape(-1, 2)

#outputFeatureMatrix = np.array(np.meshgrid(outputListOfUsersNumpy, outputListOfMoviesNumpy))
#outputFeatureMatrixNumpy = outputFeatureMatrix.T.reshape(-1, 2)

def makeFileDict(path):
        '''
    Returns a nested default dictionary (we need to set default as 0)
        '''
        trainingFile = open(path)
        inputFileDict = defaultdict(lambda: defaultdict(float))
        
        for line in trainingFile:
            row = int(line.split(',')[1])
            column = int(line.split(',')[0])
            key = float(line.split(',')[2].strip())
            inputFileDict[row][column] = key
        return inputFileDict
    
inputFileDict = makeFileDict("C:/Users/Friday/Desktop/F21 Books/CS 6375/Homework 2/TrainingRatings_Sample.txt")
#outputFileDict = makeFileDict("C:/Users/Friday/Desktop/F21 Books/CS 6375/Homework 2/TestingRatings.txt")

#Enter appropriate rating to the combination which are available in training data. 0 for the rest.
inputListOfRatings = []
for count in range(len(inputFeatureMatrixNumpy)):
   if(inputFileDict[inputFeatureMatrixNumpy[count][0]][inputFeatureMatrixNumpy[count][1]]):
       #np.insert(inputFeatureMatrixNumpy[count], 2, inputFileDict[inputFeatureMatrixNumpy[count][0]][inputFeatureMatrixNumpy[count][1]])
       inputListOfRatings.append(inputFileDict[inputFeatureMatrixNumpy[count][0]][inputFeatureMatrixNumpy[count][1]])
   else:
       inputListOfRatings.append(0.0)

inputListOfRatings = np.array(inputListOfRatings)
#where -1 infers the size of the new dimension from the size of the input array.
#inputListOfRatings = np.reshape(inputListOfRatings, (-1, 52768938))
inputListOfRatings = np.reshape(inputListOfRatings, (-1, 157190))

#At this point, inputFeatureMatrixNumpy is of (28978 * 1821, 3) dimensions
inputFeatureMatrixNumpy = np.concatenate((inputFeatureMatrixNumpy, inputListOfRatings.T), axis=1)

# outputListOfRatings = []
# for count in range(len(outputFeatureMatrixNumpy)):
#    if(outputFileDict[outputFeatureMatrixNumpy[count][0]][outputFeatureMatrixNumpy[count][1]]):
#        outputListOfRatings.append(outputFileDict[outputFeatureMatrixNumpy[count][0]][outputFeatureMatrixNumpy[count][1]])
#    else:
#        outputListOfRatings.append(0.0)

# outputListOfRatings = np.array(outputListOfRatings)
# outputListOfRatings = np.reshape(outputListOfRatings, (-1, 52768938))
# outputFeatureMatrixNumpy = np.concatenate((outputFeatureMatrixNumpy, outputListOfRatings.T), axis=1)

#rms = mean_squared_error(outputListOfRatings, inputListOfRatings,  squared=False)
#print(rms)

dfInput = pd.DataFrame(inputFeatureMatrixNumpy, columns = ['Users', 'Movies', 'Ratings'])
#dfOutput = pd.DataFrame(outputFeatureMatrixNumpy, columns = ['Users', 'Movies', 'Ratings'])
#print(dfInput.head(10))

dfMeanVote = dfInput.copy().drop(columns='Movies')
#Converting 0.0 ratings to NaN. Previously 0.0 was being considered while computing mean
#dfMeanVote = dfMeanVote.replace(0.0, np.NaN)
dfMeanVote = dfMeanVote.groupby(['Users'], as_index=False, sort=False).mean().rename(columns = {'Ratings':'MeanRatings'})
dfInput = pd.merge(dfInput, dfMeanVote, on='Users', how='left', sort=False)
dfInput['RatingsCentered'] = dfInput['Ratings'] - dfInput['MeanRatings']

dfInputMatrix = pd.DataFrame({"Users":dfInput['Users'],
                            "Movies":dfInput['Movies'],
                            "Ratings":dfInput["RatingsCentered"]})
dfInputMatrix = dfInputMatrix.pivot_table(index='Users',columns='Movies',values='Ratings').fillna(0)

print(dfInputMatrix.head(20))

#print(dfMeanVote.head(10))


stop = timeit.default_timer()
print("Runtime  = ", stop-start)

#THIS CODE FOR PANDAS 

# users = dfInputMatrix.to_numpy()
# eligibleUsers = np.array([])
# for i in range(len(users)):
#     eligibility = np.count_nonzero(users[i])
#     if(eligibility >= 100):
#         eligibleUsers = np.append(eligibleUsers, dfInput.index[i])

# weightMatrix = np.array([]) 
# tempCount = 1
# for a in range(len(users[:1000])):
#     activeUser = users[a]
#     # eligibility = np.count_nonzero(activeUser)
#     # if(eligibility >= 20):
#     denominatorPart1 = np.sqrt(sum([np.square(rating) for rating in activeUser]))
#     similarity = np.array([])
#     #pointer = 1
#     for user in users[:1000]:
#     #if(user != activeUser):
#         numerator = [movieActiveUser*movieUser for movieActiveUser, movieUser in zip(activeUser, user)]
#         denominatorPart2 = np.sqrt(sum([np.square(rating) for rating in user]))
#         if(denominatorPart1 != 0 and denominatorPart2 != 0):
#             cosineValue = sum(numerator) / (denominatorPart1*denominatorPart2)
#         #similarity.append((dfInputMatrix.index[pointer], cosineValue))
#             similarity = np.append(similarity, cosineValue)
#     if tempCount == 1:
#         weightMatrix = similarity
#         tempCount = 0
#     else:
#         weightMatrix = np.vstack((weightMatrix, similarity))

#cosineValue.sorted(key=lambda x: x[1], reverse=True)

#similar10Users = similarity[0:10]
#print(weightMatrix.head)

# dfWeightMatrix = pd.DataFrame({"UsersRow":dfInput['Users'],
#                             "UsersCol":dfInput['Users'],
#                             "Weights":weightMatrix})
# dfWeightMatrix = dfWeightMatrix.pivot_table(index='UsersRow',columns='UsersCol',values='Weights')

# weights = weightMatrix.values

#Forming a matrix (dimensions - users x movies) with ZeroCentered ratings (rawRating - MeanRating)
# dfInputMatrixCenteredRatings = pd.DataFrame({"Users":dfInput['Users'],
#                             "Movies":dfInput['Movies'],
#                             "Ratings":dfInput["RatingsCentered"]})
# dfInputMatrixCenteredRatings = dfInputMatrixCenteredRatings.pivot_table(index='Users',columns='Movies',values='Ratings').fillna(0)

#print(weightMatrixNumpy)
#listOfUsers = np.asarray(dfInputMatrixRawRatings.index)
#listOfMovies = np.asarray(dfInputMatrixRawRatings.columns)
#Computing cosine similarity between two users. A value close to 1 indicates higher similarity. Hence better 
#potential to be considered a neighbor

# cosineSimilarity = cosine_similarity(dfInputMatrixRawRatings)
# print(cosine_similarity)

# # def calculateSimilarity(activeUser, user, i):
# #     '''
# #     Calculate similarity between users using Correlation
# #     '''
    
# #     #Both active user and user need to vote on movie j
# #     numerator1, numerator2, denominator1, denominator2 = 0, 0, 0, 0
# #     #find index of user. Think to use Hashmap for these two listOfMovies and listOfUsers
# #     for users in range(len(listOfUsers)):
# #         if(listOfUsers[users] == user):
# #             indexOfUser = users
# #     for movies in listOfMovies:
# #         if(dfInputMatrixRawRatings.loc[activeUser][movies] > 0 and dfInputMatrixRawRatings.loc[user][movies] > 0):
# #             numerator1 += dfInputMatrixRawRatings.loc[activeUser][movies] - dfInput['MeanRatings'].loc[dfOutput['Users'][i]]
# #             numerator2 += dfInputMatrixRawRatings.loc[user][movies] - dfInput['MeanRatings'].loc[dfOutput['Users'][indexOfUser]]
# #             denominator1 += math.pow(dfInputMatrixRawRatings.loc[activeUser][movies] - dfInput['MeanRatings'].loc[dfOutput['Users'][i]], 2)
# #             denominator2 += math.pow(dfInputMatrixRawRatings.loc[user][movies] - dfInput['MeanRatings'].loc[dfOutput['Users'][indexOfUser]], 2)
# #     numerator = numerator1 * numerator2
# #     denominator = denominator1 * denominator2
# #     return (numerator/denominator)


#collectedWeight += dfWeightMatrix.loc[dfOutput['Users'][i], user]*dfInputMatrixCenteredRatings.loc[user][dfOutput['Movies'][i]]
                #weight = calculateSimilarity(dfOutput['Users'][i], user, i)
                #collectedWeight += weight*(dfInputMatrixRawRatings.loc[user][dfOutput['Movies'][i]] - dfInput['MeanRatings'].loc[dfOutput['Users'][i]])
                