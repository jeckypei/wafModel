import os
import pprint
import pickle
from pathlib import Path
import glob
import json
from io import open
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from transformers import AutoTokenizer
from tokenizers import Tokenizer

trainWeight=0.5
validWeight=0.25
dataLoaded = False
forceParse = False 
maxMaliciousLen = 0
maxSeqLen = -1

classId2NameList = []
maliciousFnameList = []
targetList = [] 
className2IdDict = {}
classId2NameDict = {}

trainDataSet = [] 
trainTargetSet = [] 
valDataSet = []
valTargetSet = []
testDataSet = []
testTargetSet = []

# Specify the directory path
maliciousDir = './data/Malicious/'
legitimateDir = './data/Legitimate/'

classId2NameList.append("normal")
className2IdDict['normal'] = 0
classId2NameDict[0] = 'normal' 
classDataNumbers = {} 
maliciousTrainNum = 0
trained_tokenizer = Tokenizer.from_file("./tokenizer.json")
def parseJsonFile(fname):
  with open(fname, 'r', encoding='utf-8') as f:
    data = json.load(f)
  return data
    
def formatJsonData(obj):
  return json.dumps(obj)  
   
def parseLegitimate():
  global targetList
  global trainDataSet 
  global trainTargetSet 
  global valDataSet 
  global valTargetSet 
  global testDataSet 
  global testTargetSet 
  global maliciousTrainNum
  global classDataNumbers
  fileList = glob.glob(legitimateDir + "/*")
  fNum = len(fileList)
  index = 0
  for fname in fileList:
    index = index + 1
    if (index > 32):
        break
    print("parsing (%d / %d) file: %s"  %(index, fNum, fname))
    jsonData = parseJsonFile(fname)
    print("  items num: %d" %(len(jsonData)))
    #split it
    train_size = int(trainWeight * len(jsonData))
    val_size = int(validWeight * len(jsonData))
    test_size = len(jsonData) - train_size - val_size
    
    #tokenize
    tokenData = []
    for item in jsonData:
      tmpData = formatJsonData(item)
      tokenIds = trained_tokenizer.encode(tmpData).ids
      tokenData.append(tokenIds)
    
    print("  split count: train:%d val:%d test %d" %(train_size, val_size, val_size))
    train_dataset, val_dataset, test_dataset = random_split(tokenData, [train_size, val_size, test_size])
    classDataNumbers['normal']['total'] += len(jsonData)
    classDataNumbers['normal']['train'] += len(train_dataset)
    classDataNumbers['normal']['valid'] += len(val_dataset)
    classDataNumbers['normal']['test'] += len(test_dataset)
    
    trainList = list(train_dataset)
    trainDataSet.extend(trainList)
    for i in trainList:
      trainTargetSet.append((0))

    valList = list(val_dataset)
    valDataSet.extend(valList)
    for i in valList:
      valTargetSet.append((0))
    
    testList = list(test_dataset)  
    testDataSet.extend((test_dataset))
    for i in testList:
      testTargetSet.append((0))

    print("  trainDataset:%d trainTargeSet:%d"%(len(trainDataSet),  len(trainTargetSet)))
    print("  valDataset:%d valTargeSet:%d"%(len(valDataSet), len(valTargetSet)))
    print("  testDataset:%d testTargeSet:%d"%(len(testDataSet), len(testTargetSet)))
    if (len(trainDataSet) >= 2 * maliciousTrainNum):
        return

def initClassDataNumbers(nameList):
    global classDataNumbers
    for item in nameList:
        cnts = {}
        cnts['total'] = 0
        cnts['train'] = 0
        cnts['valid'] = 0
        cnts['test'] = 0
        classDataNumbers[item] = cnts 


def parseMalicious():
  # Open the directory and view its contents
  global targetList
  global classId2NameList 
  global maliciousFnameList 
  global trainDataSet 
  global trainTargetSet 
  global valDataSet 
  global valTargetSet 
  global testDataSet 
  global testTargetSet 
  global maxMaliciousLen
  global maliciousTrainNum
  global classDataNumbers

  contents = os.listdir(maliciousDir)
  i = 1
  #init class List and Dict
  for fname in contents:
    result = fname.split('.')
    classId2NameList.append(result[0])
    className2IdDict[result[0]] = i
    classId2NameDict[i] = result[0] 
    maliciousFnameList.append(fname)
    i = i + 1
  
  initClassDataNumbers(classId2NameList)
    
  #init target list
  num = len(classId2NameList)
  #it is a identity unit matrix
  print("classifications number: %d" %(num))
  #print(targetList)

  #parse every file  
  fNum = len(maliciousFnameList)
  for index,fname in enumerate(maliciousFnameList):
    ret = fname.split('.')
    cName = ret[0]
    print("parsing (%d / %d) file: %s"  %(index+1, fNum, fname))
    jsonData = parseJsonFile(maliciousDir + fname)
    #split it
    train_size = int(trainWeight * len(jsonData))
    val_size = int(validWeight * len(jsonData))
    test_size = len(jsonData) - train_size - val_size
    
    #tokenize
    tokenData = []
    for item in jsonData:
      tmpData = formatJsonData(item)
      tokenIds = trained_tokenizer.encode(tmpData).ids
      tokenData.append(tokenIds)
      sz = len(tokenIds)
      if (sz > maxMaliciousLen):
          maxMaliciousLen = sz
    
    print("  split count: train:%d val:%d test %d" %(train_size, val_size, val_size))
    train_dataset, val_dataset, test_dataset = random_split(tokenData, [train_size, val_size, test_size])
    classDataNumbers[cName]['total'] += len(jsonData)
    classDataNumbers[cName]['train'] += len(train_dataset)
    classDataNumbers[cName]['valid'] += len(val_dataset)
    classDataNumbers[cName]['test'] += len(test_dataset)
    
    trainList = list(train_dataset)
    trainDataSet.extend(trainList)
    for i in trainList:
      trainTargetSet.append((index + 1))

    valList = list(val_dataset)
    valDataSet.extend(valList)
    for i in valList:
      valTargetSet.append((index + 1))
    
    testList = list(test_dataset)  
    testDataSet.extend((test_dataset))
    for i in testList:
      testTargetSet.append((index + 1))

    maliciousTrainNum = len(trainDataSet)

    print("  maxMaliciousLen: %d" %(maxMaliciousLen))
    print("  target:")
    print(targetList)
    print("  malicious trainDataset:%d trainTargeSet:%d"%(len(trainDataSet),  len(trainTargetSet)))
    print("  valDataset:%d valTargeSet:%d"%(len(valDataSet), len(valTargetSet)))
    print("  testDataset:%d testTargeSet:%d"%(len(testDataSet), len(testTargetSet)))

def get_var_name(var):
    for name, value in globals().items():
        if value is var:
            return name 
def get_var_value(var):
    for name, value in globals().items():
        if value is var:
            return value 

def saveVar(var):
    with open("./data/persist/" + get_var_name(var)+".pkl", 'wb') as file:
        pickle.dump(var, file)

def saveJson(var):
    with open("./data/persist/" + get_var_name(var)+".json", 'w') as file:
        json.dump(var, file)

        
def loadVar(var):
  with open("./data/persist/" + get_var_name(var)+".pkl", 'rb') as file:
     print("loading var " + get_var_name(var))
     if type(var) == int: 
        var = pickle.load(file)
     else:
        var.extend(pickle.load(file))
  return var

def loadJson(var):
  with open("./data/persist/" + get_var_name(var)+".json", 'r') as file:
     print("loading var " + get_var_name(var))
     var = json.load(file)
     return var

def saveVars():
    global trainDataSet 
    global trainTargetSet 
    global valDataSet 
    global valTargetSet 
    global testDataSet 
    global testTargetSet 
    global maxMaliciousLen 
    global className2IdDict
    global classId2NameDict
    global classDataNumbers
    saveVar(maxMaliciousLen)
    saveVar(trainDataSet)
    saveVar(trainTargetSet)
    saveVar(valDataSet)
    saveVar(valTargetSet)
    saveVar(testDataSet)
    saveVar(testTargetSet)
    saveJson(className2IdDict)
    saveJson(classId2NameDict)
    saveJson(classDataNumbers)
     

def initMaxSeqLen():
    global trainDataSet 
    global valDataSet 
    global testDataSet 

    global maxSeqLen
    global maxMaliciousLen
    maxSeqLen = maxMaliciousLen
    #if(maxSeqLen > 1024):
    #    maxSeqLen = 1024
    '''
    for data in trainDataSet:
        sz = len(data)
        if (sz > maxSeqLen):
            maxSeqLen = sz 
    for data in valDataSet:
        sz = len(data)
        if (sz > maxSeqLen):
            maxSeqLen = sz 
    for data in testDataSet:
        sz = len(data)
        if (sz > maxSeqLen):
            maxSeqLen = sz 
    '''

def loadVars():
    global trainDataSet 
    global trainTargetSet 
    global valDataSet 
    global valTargetSet 
    global testDataSet 
    global testTargetSet 
    global maxMaliciousLen 
    global className2IdDict
    global classId2NameDict
    global classDataNumbers
    maxMaliciousLen = loadVar(maxMaliciousLen)
    loadVar(trainDataSet)
    loadVar(trainTargetSet)
    loadVar(valDataSet)
    loadVar(valTargetSet)
    loadVar(testDataSet)
    loadVar(testTargetSet)
    className2IdDict = loadJson(className2IdDict)
    classId2NameDict = loadJson(classId2NameDict)
    classDataNumbers = loadJson(classDataNumbers)

def loadDataSet():
    global maxSeqLen
    global forceParse
    if forceParse:
        parseMalicious()
        parseLegitimate()
        saveVars()
    else:
        loadVars()
    initMaxSeqLen()

def getDataSet():
    global dataLoaded
    if not dataLoaded:
        loadDataSet()
        dataLoaded = True 
    return (trainDataSet, trainTargetSet, valDataSet, valTargetSet, testDataSet, testTargetSet, maxSeqLen)


class Corpus(object):
    def __init__(self):
        global trainDataSet 
        global trainTargetSet 
        global valDataSet 
        global valTargetSet 
        global testDataSet 
        global testTargetSet 
        global maxSeqLen
        global className2IdDict
        global classId2NameDict
        global classDataNumbers 

        loadDataSet()
        self.formatData()
        self.train = torch.tensor(trainDataSet)  
        self.trainTarget = torch.tensor(trainTargetSet)  
        tokenizer = Tokenizer.from_file("./tokenizer.json")
        self.vocab_size = tokenizer.get_vocab_size()
        print("vocab_size %d" %self.vocab_size)
        print("train info ")
        print(self.train.shape)
        print(self.trainTarget.shape)

        self.valid = torch.tensor(valDataSet) 
        self.validTarget = torch.Tensor(valTargetSet) 
        print("valid info ")
        print(self.valid.shape)
        print(self.validTarget.shape)

        self.test = torch.tensor(testDataSet) 
        self.testTarget = torch.tensor(testTargetSet) 
        print("test info ")
        print(self.test.shape)
        print(self.testTarget.shape)

        self.maxSeqLen = maxSeqLen
        self.className2IdDict = className2IdDict
        self.classId2NameDict = classId2NameDict
        self.classNameList = [None] * len(self.classId2NameDict) 
        for key, item in self.classId2NameDict.items():
            self.classNameList[int(key)] = item
        print("Data weights: train %.2f, valid %.2f, test %.2f" %(trainWeight, validWeight, (1 - trainWeight - validWeight)))
        print("Train data num: %d, Validation data num:%d, Test data num:%d" %(self.train.size(0), self.valid.size(0), self.test.size(0)))
        print("\nDetail data numbers:")
        pprint.pprint(classDataNumbers)

    def formatData(self): 
        global trainDataSet 
        global trainTargetSet 
        global valDataSet 
        global valTargetSet 
        global testDataSet 
        global testTargetSet 
        global maxSeqLen
        print("maxSeqLen %d" %(maxSeqLen))

        for index, item in enumerate(trainDataSet):
            if len(item) >= maxSeqLen:
                trainDataSet[index] = item[:maxSeqLen]
            else:
                pad = [0] * (maxSeqLen - len(item))
                trainDataSet[index] = item + pad 

        for index, item in enumerate(valDataSet):
            if len(item) >= maxSeqLen:
                valDataSet[index] = item[:maxSeqLen]
            else:
                pad = [0] * (maxSeqLen - len(item))
                valDataSet[index] = item + pad

        for index, item in enumerate(testDataSet):
            if len(item) >= maxSeqLen:
                testDataSet[index] = item[:maxSeqLen]
            else:
                pad = [0] * (maxSeqLen - len(item))
                testDataSet[index] = item + pad


if __name__ == "__main__": 
    corpus = Corpus()
