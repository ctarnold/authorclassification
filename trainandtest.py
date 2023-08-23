import os
import pandas as pd
import math

directory = "C:/Users/chris/OneDrive/Desktop/Academic/23Fall/JuniorWork/author classification/trainingspreadsheets"

i = 0

vocabularyset = set()
LAUcount = 0
CHLcount = 0

LAUwordmap = {}
CHLwordmap = {}

# build a vocabulary set of the training data
# build wordmaps with word -> count for each speaker
for filename in os.listdir(directory):
    if filename.endswith("zeledon3.tsv"): 
        df = pd.read_csv(directory + "/" + filename, usecols = ['surface', 'speaker'], sep = '\t')
        for row in df.iterrows():
            word = row[1][0]
            vocabularyset.add(word)
            if (row[1][1] == "ELE"):
                LAUcount +=1
                if (LAUwordmap.__contains__(word)):
                    count = LAUwordmap.get(word)
                    LAUwordmap[word] = count + 1
                else:
                    LAUwordmap[word] = 1
            if (row[1][1] == "FEL"):
                CHLcount+=1
                if (CHLwordmap.__contains__(word)):
                    count = CHLwordmap.get(word)
                    CHLwordmap[word] = count + 1
                else:
                    CHLwordmap[word] = 1
            i +=1

    else:
        continue
print(len(vocabularyset))
print(i)
print(LAUcount)
print(CHLcount)

classes = ["LAU", "CHL"]
classCounts = [LAUcount, CHLcount]
logprior = [0, 0]

wordMaps = [LAUwordmap, CHLwordmap]

loglikelihood = {}

index = 0
for val in classes:
    
    Ndoc = i
    Nclass = CHLcount
    if (classes[index] == "LAU"):
        Nclass = LAUcount
    logprior[index] = math.log(Nclass/Ndoc)
   
    for word in vocabularyset:
        count = int(wordMaps[index].get(word) or 0)
        numerator = count + 1
        denominator = vocabularyset.__sizeof__()
        denominator += classCounts[index]
        prelimVal = numerator/denominator
        loglikelihood[word, classes[index]] = math.log(prelimVal)
    index+=1

# Test Naive Bayes



# first separate the testing csv data into a set of
# sentences for each speaker
LAUsentences = set()
CHLsentences = set()
sentences = set()

directory = "C:/Users/chris/OneDrive/Desktop/Academic/23Fall/JuniorWork/author classification/testingspreadsheets"

lastSpeaker = 'ELE'
# build a set of sentences in the testing data for each speaker
for filename in os.listdir(directory):
    runningString = ""
    if(filename.endswith("zeledon3test.csv")):
        df = pd.read_csv(directory + "/" + filename, 
                         usecols = ['surface', 'speaker'], sep = ',')
       
        for row in df.iterrows():
            # print(row[1][1])
            if (row[1][1] == lastSpeaker):
                runningString = runningString + (" " + row[1][0])
            else: 
                # print(runningString + "\n")
                if (lastSpeaker == 'ELE'):
                   # print("reached")
                    LAUsentences.add(runningString)
                 
                if (lastSpeaker == 'FEL'):
                    #print("reached")
                    CHLsentences.add(runningString)
                   

                if (lastSpeaker == 'FEL'):
                    lastSpeaker = 'ELE'
                else:
                    lastSpeaker = 'FEL'
                sentences.add(runningString)
                runningString = row[1][0]




sum = [0,0]
index = 0

correctCount = 0
count = 0
skipped = 0

sentenceEvaluated = False

for sentence in sentences:
    sentenceEvaluated = False
    if (CHLsentences.__contains__(sentence)):
        if (LAUsentences.__contains__(sentence)):
            skipped = skipped + 1
            continue
    prediction = 'ELE'
    answer = 'ELE'
    wordlist = sentence.split()
    for word in wordlist:
        if (vocabularyset.__contains__(word)):
            sentenceEvaluated = True
            sum[1] = sum[1] + loglikelihood[word, classes[1]]
            sum[0] = sum[0] + loglikelihood[word, classes[0]]
    if (sentenceEvaluated):
        if (sum[1] > sum[0]):
            prediction = 'FEL'
        if (CHLsentences.__contains__(sentence)):
            answer = 'FEL'

        if (prediction == answer):
            correctCount = correctCount + 1
        count = count + 1
    
print(correctCount)
print(count)
print(skipped)
print(correctCount/count)
print("above is the accuracy for individual sentences")


# what if I wanted to take a group of 4 sentences, would that improve the 
# accuracy?

LAUparagraphs = set()
CHLparagraphs = set()
paragraphs = set()

stringBuilder = ""

count = 1
divFactor = 4
for sentence in LAUsentences:
    stringBuilder = stringBuilder + " " + sentence
    if (count % divFactor == 0):
        LAUparagraphs.add(stringBuilder)
        paragraphs.add(stringBuilder)
        stringBuilder = ""
    count = count + 1

count = 1
stringBuilder = ""
for sentence in CHLsentences:
    stringBuilder = stringBuilder + " " + sentence
    if (count % divFactor == 0):
        CHLparagraphs.add(stringBuilder)
        paragraphs.add(stringBuilder)
        stringBuilder = ""
    count = count + 1


correctCount = 0
count = 0
skipped = 0

sentenceEvaluated = False

for paragraph in paragraphs:
    sentenceEvaluated = False
    prediction = 'ELE'
    answer = 'ELE'
    wordlist = paragraph.split()
    for word in wordlist:
        if (vocabularyset.__contains__(word)):
            sentenceEvaluated = True
            sum[1] = sum[1] + loglikelihood[word, classes[1]]
            sum[0] = sum[0] + loglikelihood[word, classes[0]]
    if (sentenceEvaluated):
        if (sum[1] > sum[0]):
            prediction = 'FEL'
        if (CHLparagraphs.__contains__(paragraph)):
            answer = 'FEL'

        if (prediction == answer):
            correctCount = correctCount + 1
        count = count + 1
    
print(correctCount)
print(count)
print(skipped)
print(correctCount/count)
print("above is the accuracy for collections of " + divFactor.__str__() + " sentences")
