# -*- coding: utf-8 -*-

import numpy as np

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

#setup arrays for loop use
femaleScores = []
maleScores = []
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

#extract female and male scores
for index, item in enumerate(sex):
    if item == "Female":
        femaleScores.append(healthScore[index])
    elif item == "Male":
        maleScores.append(healthScore[index])
        
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

print (min(iq))
print (max(iq))

#print (age)
#print (sex)
#print (weight)
#print (height)
#print (iq)
#print (unitsAlcohol)
#print (unitsCigs)
#print (activity)
#print (healthScore)