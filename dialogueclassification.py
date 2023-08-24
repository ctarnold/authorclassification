import os
import pandas as pd
import math

directory = "local dir omitted"

i = 0

groupFactor = 5

vocabularyset = set()
HERcount = 0
ZELcount = 0

HERsentencecount = 0
ZELsentencecount = 0

HERwordmap = {}
ZELwordmap = {}

# build a vocabulary set of the training data for herring1train and zeledon3
# build wordmaps with word -> count for each herring
for filename in os.listdir(directory):
    if filename.endswith("herring1.tsv"): 
        df = pd.read_csv(directory + "/" + filename, usecols = ['surface', 'speaker'], sep = '\t')
        for row in df.iterrows():
            HERcount = HERcount + 1
            word = row[1][0]
            vocabularyset.add(word)
            if (HERwordmap.__contains__(word)):
                count = HERwordmap.get(word)
                HERwordmap[word] = count + 1
            else:
                HERwordmap[word] = 1
            if (word == '.' or word == '!'):
                HERsentencecount += 1
    if filename.endswith("zeledon3.tsv"): 
        df = pd.read_csv(directory + "/" + filename, usecols = ['surface', 'speaker'], sep = '\t')
        for row in df.iterrows():
            ZELcount = ZELcount + 1
            word = row[1][0]
            vocabularyset.add(word)
            if (ZELwordmap.__contains__(word)):
                count = ZELwordmap.get(word)
                ZELwordmap[word] = count + 1
            else:
                ZELwordmap[word] = 1
            if (word == '.' or word == '!' or word == '?'):
                ZELsentencecount += 1
print(len(vocabularyset))
print(HERcount)
print(ZELcount)

classes = ["HER", "ZEL"]
classCounts = [HERcount, ZELcount]
logprior = [0, 0]
logpriorGrouped = [0, 0]

wordMaps = [HERwordmap, ZELwordmap]

loglikelihood = {}
loglikelihoodGrouped = {}

# Train Naive Bayes
index = 0
for val in classes:
    
    Ndoc = HERsentencecount + ZELsentencecount
    Nclass = HERsentencecount
    if (classes[index] == "ZEL"):
        Nclass = ZELsentencecount
    logprior[index] = math.log(Nclass/Ndoc)
    logpriorGrouped[index] = math.log((Nclass/groupFactor)/ (Ndoc/groupFactor))
   
    for word in vocabularyset:
        count = int(wordMaps[index].get(word) or 0)
        numerator = count + 1
        denominator = 0
        for word2 in vocabularyset:
            if (wordMaps[index].__contains__(word2)):
                denominator = denominator + wordMaps[index].get(word2) + 1
        # denominator = vocabularyset.__sizeof__()
        # denominator += classCounts[index]
        prelimVal = numerator/denominator
        loglikelihood[word, classes[index]] = math.log(prelimVal)
    index+=1



# Test Naive Bayes

# first separate the testing csv data into a set of
# sentences for each dialogue
HERsentenceSet = set()
HERparagraphSet = set()
ZELsentenceSet = set()
ZELparagraphSet = set()

sentences = set()
paragraphs = set()

counter = 1

directory = "local dir omitted"

lastSpeaker = 'ELE'
# build a set of sentences in the testing data for each dialogue
for filename in os.listdir(directory):
    runningString = ""
    runningParagraph = ""
    if(filename.endswith("zeledon3test.csv")):
        df = pd.read_csv(directory + "/" + filename, 
                         usecols = ['surface', 'speaker'], sep = ',')
       
        for row in df.iterrows():
            word = row[1][0]
            if (word == '.' or word == '!' or word == '?'):
                ZELsentenceSet.add(runningString)
                sentences.add(runningString)
                runningParagraph = runningParagraph + " " + runningString
                if (counter % groupFactor == 0):
                    ZELparagraphSet.add(runningParagraph)
                    paragraphs.add(runningParagraph)
                    runningParagraph = ""
                runningString = ""   
            else:
                runningString = runningString + " " + word
            counter += 1
    runningString = ""
    counter = 1
    if(filename.endswith("herring1test1.csv")):
        df = pd.read_csv(directory + "/" + filename, 
                         usecols = ['surface', 'speaker'], sep = ',')
       
        for row in df.iterrows():
            word = row[1][0]
            if (word == '.' or word == '!' or word == '?'):
               HERsentenceSet.add(runningString)
               sentences.add(runningString)
               runningParagraph = runningParagraph + " " + runningString
               if (counter % groupFactor == 0):
                    HERparagraphSet.add(runningParagraph)
                    paragraphs.add(runningParagraph)
                    runningParagraph = ""
               runningString = ""
            else:
                runningString = runningString + " " + word
            counter += 1


sum = [logprior[0], logprior[1]]
index = 0

correctCount = 0
count = 0
skipped = 0

sentenceEvaluated = False

for sentence in sentences:
    sum = [logprior[0], logprior[1]]
    wordlist = sentence.split()
    for word in wordlist:
        if (vocabularyset.__contains__(word)):
            sum[0] = sum[0] + loglikelihood[word, classes[0]]
            sum[1] = sum[1] + loglikelihood[word, classes[1]]
        else:
            skipped += 1
    answer = "HER"
    prediction = "HER"
    if (ZELsentenceSet.__contains__(sentence)):
        answer = "ZEL"
    if (sum[1] > sum[0]):
        prediction = "ZEL"
    if (answer == prediction):
        correctCount += 1
    count += 1
    



print("\n")
print(correctCount)
print(count)
print(skipped)
print(correctCount/count)
print("above is the accuracy for individual sentences")

index = 0

correctCount = 0
count = 0
skipped = 0

sentenceEvaluated = False

for sentence in paragraphs:
    sum = [logpriorGrouped[0], logpriorGrouped[1]]
    wordlist = sentence.split()
    for word in wordlist:
        if (vocabularyset.__contains__(word)):
            sum[0] = sum[0] + loglikelihood[word, classes[0]]
            sum[1] = sum[1] + loglikelihood[word, classes[1]]
        else:
            skipped += 1
            # print(word + " skipped")
    answer = "HER"
    prediction = "HER"
    if (ZELparagraphSet.__contains__(sentence)):
        answer = "ZEL"
    if (sum[1] > sum[0]):
        prediction = "ZEL"
    if (answer == prediction):
        correctCount += 1
    count += 1

print("\n")
print(correctCount)
print(count)
print(skipped)
print(correctCount/count)
print("above is the accuracy for groups of sentences")




# what if I wanted to take a group of 4 sentences, would that improve the 
# accuracy?

# LAUparagraphs = set()
# CHLparagraphs = set()
# paragraphs = set()

# stringBuilder = ""

# count = 1
# divFactor = 4
# for sentence in LAUsentences:
#     stringBuilder = stringBuilder + " " + sentence
#     if (count % divFactor == 0):
#         LAUparagraphs.add(stringBuilder)
#         paragraphs.add(stringBuilder)
#         stringBuilder = ""
#     count = count + 1

# count = 1
# stringBuilder = ""
# for sentence in CHLsentences:
#     stringBuilder = stringBuilder + " " + sentence
#     if (count % divFactor == 0):
#         CHLparagraphs.add(stringBuilder)
#         paragraphs.add(stringBuilder)
#         stringBuilder = ""
#     count = count + 1


# correctCount = 0
# count = 0
# skipped = 0

# sentenceEvaluated = False

# for paragraph in paragraphs:
#     sentenceEvaluated = False
#     prediction = 'ELE'
#     answer = 'ELE'
#     wordlist = paragraph.split()
#     for word in wordlist:
#         if (vocabularyset.__contains__(word)):
#             sentenceEvaluated = True
#             sum[1] = sum[1] + loglikelihood[word, classes[1]]
#             sum[0] = sum[0] + loglikelihood[word, classes[0]]
#     if (sentenceEvaluated):
#         if (sum[1] > sum[0]):
#             prediction = 'FEL'
#         if (CHLparagraphs.__contains__(paragraph)):
#             answer = 'FEL'

#         if (prediction == answer):
#             correctCount = correctCount + 1
#         count = count + 1
    
# print(correctCount)
# print(count)
# print(skipped)
# print(correctCount/count)
# print("above is the accuracy for collections of " + divFactor.__str__() + " sentences")
