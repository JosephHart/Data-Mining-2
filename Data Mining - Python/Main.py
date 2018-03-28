# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans

#Read file in
dataArray = np.genfromtxt('HealthScores.csv', delimiter=',', dtype=None).astype(str)

#Extract variables from raw input
age = [dataArray[i][0] for i in range(len(dataArray))]
sex = [dataArray[i][1] for i in range(len(dataArray))]
weight = [dataArray[i][2] for i in range(len(dataArray))]
height = [dataArray[i][3] for i in range(len(dataArray))]
iq = [dataArray[i][4] for i in range(len(dataArray))]
unitsAlcohol = [dataArray[i][5] for i in range(len(dataArray))]
unitsCigs = [dataArray[i][6] for i in range(len(dataArray))]
activity = [dataArray[i][7] for i in range(len(dataArray))]
healthScore = [dataArray[i][8] for i in range(len(dataArray))]

#strip off column labels
del age[0]
del sex[0]
del weight[0]
del height[0]
del iq[0]
del unitsAlcohol[0]
del unitsCigs[0]
del activity[0]
del healthScore[0]

#convert to ints where appropriate
age = list(map(int, age))
weight = list(map(int, weight))
height = list(map(int, height))
iq = list(map(int, iq))
unitsAlcohol = list(map(int, unitsAlcohol))
unitsCigs = list(map(int, unitsCigs))
healthScore = list(map(int, healthScore))

for index, item in enumerate(unitsAlcohol):
    unitsAlcohol[index] = 0

for index, item in enumerate(unitsCigs):
    unitsCigs[index] = 0

#setup arrays for loop use
femaleScores = []
maleScores = []
genderNumberScores = []

scores1830 = []
scores3140 = []
scores4150 = []
scores5160 = []
scores6170 = []
scores7180 = []

scoresWeight77 = []
scoresWeight105 = []
scoresWeight133 = []
scoresWeight161 = []
scoresWeight189 = []

scoresHeight56 = []
scoresHeight63 = []
scoresHeight70 = []
scoresHeight76 = []

scoresIQ79 = []
scoresIQ90 = []
scoresIQ101 = []
scoresIQ112 = []

scoresAlcohol0 = []
scoresAlcohol1 = []
scoresAlcohol3 = []
scoresAlcohol5 = []

scoresCigs0 = []
scoresCigs1 = []
scoresCigs11 = []
scoresCigs21 = []

scoresInactive = []
scoresActive = []
scoresVeryActive = []
scoresNumberActivity = []

#extract female and male scores
for index, item in enumerate(sex):
    if item == "Female":
        femaleScores.append(healthScore[index])
        genderNumberScores.append(0)
    elif item == "Male":
        maleScores.append(healthScore[index])
        genderNumberScores.append(1)
        
#extract age group scores
for index, item in enumerate(age):
    if int(item) >= 18 and int(item) <= 30:
        scores1830.append(healthScore[index])
    elif int(item) >= 31 and int(item) <= 40:
        scores3140.append(healthScore[index])
    elif int(item) >= 41 and int(item) <= 50:
        scores4150.append(healthScore[index])
    elif int(item) >= 51 and int(item) <= 60:
        scores5160.append(healthScore[index])
    elif int(item) >= 61 and int(item) <= 70:
        scores6170.append(healthScore[index])
    elif int(item) >= 71 and int(item) <= 80:
        scores7180.append(healthScore[index])
        
#extract weight group scores
for index, item in enumerate(weight):
    if int(item) >= 77 and int(item) <= 104:
        scoresWeight77.append(healthScore[index])
    elif int(item) >= 105 and int(item) <= 132:
        scoresWeight105.append(healthScore[index])
    elif int(item) >= 133 and int(item) <= 160:
        scoresWeight133.append(healthScore[index])
    elif int(item) >= 161 and int(item) <= 188:
        scoresWeight161.append(healthScore[index])
    elif int(item) >= 189 and int(item) <= 217:
        scoresWeight189.append(healthScore[index])
        
#extract height group scores
for index, item in enumerate(height):
    if int(item) >= 56 and int(item) <= 62:
        scoresHeight56.append(healthScore[index])
    elif int(item) >= 63 and int(item) <= 69:
        scoresHeight63.append(healthScore[index])
    elif int(item) >= 70 and int(item) <= 76:
        scoresHeight70.append(healthScore[index])
    elif int(item) >= 76 and int(item) <= 83:
        scoresHeight76.append(healthScore[index])
        
#extract iq group scores
for index, item in enumerate(iq):
    if int(item) >= 79 and int(item) <= 89:
        scoresIQ79.append(healthScore[index])
    elif int(item) >= 90 and int(item) <= 100:
        scoresIQ90.append(healthScore[index])
    elif int(item) >= 101 and int(item) <= 111:
        scoresIQ101.append(healthScore[index])
    elif int(item) >= 112 and int(item) <= 122:
        scoresIQ112.append(healthScore[index])
        
#extract drinking group scores
for index, item in enumerate(unitsAlcohol):
    if int(item) >= 0 and int(item) <= 0:
        scoresAlcohol0.append(healthScore[index])
    elif int(item) >= 1 and int(item) <= 2:
        scoresAlcohol1.append(healthScore[index])
    elif int(item) >= 3 and int(item) <= 4:
        scoresAlcohol3.append(healthScore[index])
    elif int(item) >= 5 and int(item) <= 7:
        scoresAlcohol5.append(healthScore[index])
    
#extract smoking group scores
for index, item in enumerate(unitsCigs):
    if int(item) >= 0 and int(item) <= 0:
        scoresCigs0.append(healthScore[index])
    elif int(item) >= 1 and int(item) <= 10:
        scoresCigs1.append(healthScore[index])
    elif int(item) >= 11 and int(item) <= 20:
        scoresCigs11.append(healthScore[index])
    elif int(item) >= 21 and int(item) <= 40:
        scoresCigs21.append(healthScore[index])
        
#extract activity scores
for index, item in enumerate(activity):
    if item == "Inactive":
        scoresInactive.append(healthScore[index])
        scoresNumberActivity.append(0)
    elif item == "Active":
        scoresActive.append(healthScore[index])
        scoresNumberActivity.append(1)
    elif item == "Very Active":
        scoresVeryActive.append(healthScore[index])
        scoresNumberActivity.append(2)

#print mean and median of results
print ("Mean of male scores")
print (np.mean(maleScores))
print ("Mean of female scores")
print (np.mean(femaleScores))

print ("Median of male scores")
print (np.median(maleScores))
print ("Median of female scores")
print (np.median(femaleScores))

print ("Mean of 18-30 scores")
print (np.mean(scores1830))
print ("Mean of 31-40 scores")
print (np.mean(scores3140))
print ("Mean of 41-50 scores")
print (np.mean(scores4150))
print ("Mean of 51-60 scores")
print (np.mean(scores5160))
print ("Mean of 61-70 scores")
print (np.mean(scores6170))
print ("Mean of 71-80 scores")
print (np.mean(scores7180))

print ("Median of 18-30 scores")
print (np.median(scores1830))
print ("Median of 31-40 scores")
print (np.median(scores3140))
print ("Median of 41-50 scores")
print (np.median(scores4150))
print ("Median of 51-60 scores")
print (np.median(scores5160))
print ("Median of 61-70 scores")
print (np.median(scores6170))
print ("Median of 71-80 scores")
print (np.median(scores7180))

print ("Mean of 77-104 scores")
print (np.mean(scoresWeight77))
print ("Mean of 105-132 scores")
print (np.mean(scoresWeight105))
print ("Mean of 133-160 scores")
print (np.mean(scoresWeight133))
print ("Mean of 161-188 scores")
print (np.mean(scoresWeight161))
print ("Mean of 189-217 scores")
print (np.mean(scoresWeight189))

print ("Median of 77-104 scores")
print (np.median(scoresWeight77))
print ("Median of 105-132 scores")
print (np.median(scoresWeight105))
print ("Median of 133-160 scores")
print (np.median(scoresWeight133))
print ("Median of 161-188 scores")
print (np.median(scoresWeight161))
print ("Median of 189-217 scores")
print (np.median(scoresWeight189))

print ("Mean of 56-62 scores")
print (np.mean(scoresHeight56))
print ("Mean of 63-69 scores")
print (np.mean(scoresHeight63))
print ("Mean of 70-76 scores")
print (np.mean(scoresHeight70))
print ("Mean of 76-83 scores")
print (np.mean(scoresHeight76))

print ("Median of 56-62 scores")
print (np.median(scoresHeight56))
print ("Median of 63-69 scores")
print (np.median(scoresHeight63))
print ("Median of 70-76 scores")
print (np.median(scoresHeight70))
print ("Median of 76-83 scores")
print (np.median(scoresHeight76))

print ("Mean of 79-89 scores")
print (np.mean(scoresIQ79))
print ("Mean of 90-100 scores")
print (np.mean(scoresIQ90))
print ("Mean of 101-111 scores")
print (np.mean(scoresIQ101))
print ("Mean of 112-122 scores")
print (np.mean(scoresIQ112))

print ("Median of 79-89 scores")
print (np.median(scoresIQ79))
print ("Median of 90-100 scores")
print (np.median(scoresIQ90))
print ("Median of 101-111 scores")
print (np.median(scoresIQ101))
print ("Median of 112-122 scores")
print (np.median(scoresIQ112))

print ("Mean of 0-0 scores")
print (np.mean(scoresAlcohol0))
print ("Mean of 1-2 scores")
print (np.mean(scoresAlcohol1))
print ("Mean of 3-4 scores")
print (np.mean(scoresAlcohol3))
print ("Mean of 5-7 scores")
print (np.mean(scoresAlcohol5))

print ("Median of 0-0 scores")
print (np.median(scoresAlcohol0))
print ("Median of 1-2 scores")
print (np.median(scoresAlcohol1))
print ("Median of 3-4 scores")
print (np.median(scoresAlcohol3))
print ("Median of 5-7 scores")
print (np.median(scoresAlcohol5))

print ("Mean of 0-0 scores")
print (np.mean(scoresCigs0))
print ("Mean of 1-10 scores")
print (np.mean(scoresCigs1))
print ("Mean of 11-20 scores")
print (np.mean(scoresCigs11))
print ("Mean of 21-40 scores")
print (np.mean(scoresCigs21))

print ("Median of 0-0 scores")
print (np.median(scoresCigs0))
print ("Median of 1-10 scores")
print (np.median(scoresCigs1))
print ("Median of 11-20 scores")
print (np.median(scoresCigs11))
print ("Median of 21-40 scores")
print (np.median(scoresCigs21))

print ("Mean of Inactive scores")
print (np.mean(scoresInactive))
print ("Mean of Active scores")
print (np.mean(scoresActive))
print ("Mean of Very Active scores")
print (np.mean(scoresVeryActive))

print ("Median of Inactive scores")
print (np.median(scoresInactive))
print ("Median of Active scores")
print (np.median(scoresActive))
print ("Median of Very Active scores")
print (np.median(scoresVeryActive))

print (min(unitsCigs))
print (max(unitsCigs))

#_______________________________________________________________________

from pylab import plt

#Kmeans
age = list(map(float, unitsCigs))
weight = list(map(float, healthScore))
#[age, weight, height, iq, unitsAlcohol, unitsCigs]
kMeansArray = [age, weight]
kmeans = KMeans(n_clusters=2) # initialization
kmeans.fit(kMeansArray) # actual execution
y_kmeans = kmeans.predict(kMeansArray)
centers = kmeans.cluster_centers_
print(centers)

#plt.scatter(kMeansArray[0], kMeansArray[1])

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

print(len(age))
print(len(weight))
print(len(height))
print(len(iq))
print(len(unitsAlcohol))
print(len(unitsCigs))
print(len(scoresNumberActivity))
print(len(genderNumberScores))
print(len(healthScore))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model

tempAge = age
tempWeight = weight
tempHeight = height
tempIq = iq
tempUnitsAlcohol = unitsAlcohol
tempUnitsCigs = unitsCigs
tempScoresNumberActivity = scoresNumberActivity
tempGenderNumberScores = genderNumberScores

for index, item in enumerate(age):
    item = (item-min(tempAge))/(max(tempAge)-min(tempAge))
    
for index, item in enumerate(weight):
    item = (item-min(tempWeight))/(max(tempWeight)-min(tempWeight))
    
for index, item in enumerate(height):
    item = (item-min(tempHeight))/(max(tempHeight)-min(tempHeight))
    
for index, item in enumerate(iq):
    item = (item-min(tempIq))/(max(tempIq)-min(tempIq))
    
for index, item in enumerate(unitsAlcohol):
    item = (item-min(tempUnitsAlcohol))/(max(tempUnitsAlcohol)-min(tempUnitsAlcohol))
    
for index, item in enumerate(unitsCigs):
    item = (item-min(tempUnitsCigs))/(max(tempUnitsCigs)-min(tempUnitsCigs))
    
for index, item in enumerate(tempScoresNumberActivity):
    item = (item-min(tempScoresNumberActivity))/(max(tempScoresNumberActivity)-min(tempScoresNumberActivity))
    
for index, item in enumerate(tempGenderNumberScores):
    item = (item-min(tempGenderNumberScores))/(max(tempGenderNumberScores)-min(tempGenderNumberScores))
    

x = np.array([age, genderNumberScores,  weight, height, iq, unitsAlcohol, unitsCigs, scoresNumberActivity])
y = np.array([healthScore])

print(x.shape)
print(y.shape)

x = np.transpose(x)
y = np.transpose(y)

print(x)
print(y)

print(x.shape)
print(y.shape)

predict_value = x

# Create linear regression object
regr = linear_model.LinearRegression()
regr.fit(x, y)
predict_outcome = regr.predict(predict_value)
result = {}
result['intercept'] = regr.intercept_
result['coefficient'] = regr.coef_
result['predicted_value'] = predict_outcome

print ("Intercept value " , result['intercept'])
print ("coefficient" , result['coefficient'])
print ("Predicted value: ",result['predicted_value'])

#Read file in
predictArray = np.genfromtxt('Population.csv', delimiter=',', dtype=None).astype(str)

#Extract variables from raw input
ageP = [predictArray[i][1] for i in range(len(predictArray))]
sexP = [predictArray[i][2] for i in range(len(predictArray))]
weightP = [predictArray[i][3] for i in range(len(predictArray))]
heightP = [predictArray[i][4] for i in range(len(predictArray))]
iqP = [predictArray[i][5] for i in range(len(predictArray))]
unitsAlcoholP = [predictArray[i][6] for i in range(len(predictArray))]
unitsCigsP = [predictArray[i][7] for i in range(len(predictArray))]
activityP = [predictArray[i][8] for i in range(len(predictArray))]

#strip off column labels
del ageP[0]
del sexP[0]
del weightP[0]
del heightP[0]
del iqP[0]
del unitsAlcoholP[0]
del unitsCigsP[0]
del activityP[0]

#print(activityP)

#convert to ints where appropriate
ageP = list(map(int, ageP))
weightP = list(map(int, weightP))
heightP = list(map(int, heightP))
iqP = list(map(int, iqP))
unitsAlcoholP = list(map(int, unitsAlcoholP))
unitsCigsP = list(map(int, unitsCigsP))

genderNumberScoresP = []
scoresNumberActivityP = []

#extract female and male scores
for index, item in enumerate(sexP):
    if item == "Female":
        genderNumberScoresP.append(0)
    elif item == "Male":
        genderNumberScoresP.append(1)
        
 #extract activity scores
for index, item in enumerate(activityP):
    if item == "Inactive":
        scoresNumberActivityP.append(0)
    elif item == "Active":
        scoresNumberActivityP.append(1)
    elif item == "Very Active":
        scoresNumberActivityP.append(2)
        
for index, item in enumerate(ageP):
    item = (item-min(tempAge))/(max(tempAge)-min(tempAge))
    
for index, item in enumerate(weightP):
    item = (item-min(tempWeight))/(max(tempWeight)-min(tempWeight))
    
for index, item in enumerate(heightP):
    item = (item-min(tempHeight))/(max(tempHeight)-min(tempHeight))
    
for index, item in enumerate(iqP):
    item = (item-min(tempIq))/(max(tempIq)-min(tempIq))
    
for index, item in enumerate(unitsAlcoholP):
    item = (item-min(tempUnitsAlcohol))/(max(tempUnitsAlcohol)-min(tempUnitsAlcohol))
    
for index, item in enumerate(unitsCigsP):
    item = (item-min(tempUnitsCigs))/(max(tempUnitsCigs)-min(tempUnitsCigs))
    
for index, item in enumerate(scoresNumberActivityP):
    item = (item-min(tempScoresNumberActivity))/(max(tempScoresNumberActivity)-min(tempScoresNumberActivity))
    
for index, item in enumerate(genderNumberScoresP):
    item = (item-min(tempGenderNumberScores))/(max(tempGenderNumberScores)-min(tempGenderNumberScores))
            
predictData = np.array([ageP, genderNumberScoresP, weightP, heightP, iqP, unitsAlcoholP, unitsCigsP, scoresNumberActivityP])
predictData = np.transpose(predictData)

# Create linear regression object
regrP = linear_model.LinearRegression()
regrP.fit(x, y)
predict_outcomeP = regrP.predict(predictData)
resultP = {}
resultP['predicted_value'] = predict_outcomeP
print ("Predicted value: ",resultP['predicted_value'])

print(predictData)

#show_linear_line(x,y)

#plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

#print (age)
#print (sex)
#print (weight)
#print (height)
#print (iq)
#print (unitsAlcohol)
#print (unitsCigs)
#print (activity)
#print (healthScore)