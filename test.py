import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from transformers import AutoTokenizer
import json

import data
import dataWaf
import model
import pandas as pd
from IPython.display import display

def printCnt(names, nums, failNums, sucRates, headers):
    data = {}
    data[headers[0]] = names 
    data[headers[1]] = nums 
    data[headers[2]] = failNums 
    data[headers[3]] = sucRates
    df = pd.DataFrame(data)
    display(df)


def calculateSuccessRate(classCntList, classFailCntList):
    sucRate = [] 
    for index, item in enumerate(classCntList):
        sucRate.append( 100 * ((classCntList[index] - classFailCntList[index]) / classCntList[index]))
    return sucRate

def runTestDataOneByOne():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("get system device")
    print(device)

    corpus = dataWaf.Corpus()
    testData = corpus.test
    testTarget = corpus.testTarget
    #testData.contiguous()
    testData = testData.to(device)
    testTarget.contiguous()
    testTarget = testTarget.to(device)
    #print("test shape len: data %d target %d " %(testData.size(0), testTarget.size(0)))
    print(testTarget)
    

    with open("waf-model.pt", 'rb') as f:
        model = torch.load(f)
        model.to(device)
        model.eval()

    failCnt = 0
    totalNum = testData.size(0)
    jout = {}
    allItemOut = [] 
    classCntList = [0] * len(corpus.classNameList)
    classFailCntList = [0] * len(corpus.classNameList)
    for index, item in enumerate(testData):
        itemOut = {}
        item = item.view(-1, 1)
        pred = model(item)
        pred = torch.exp(pred) 
        pred = pred.to(torch.float32)
        itemOut["index"] = index
        tgt = testTarget[index]
        classCntList[tgt] = classCntList[tgt] + 1
        if (pred[0, tgt] < 0.7):
            failCnt = failCnt + 1 
            classFailCntList[tgt] += 1
            itemOut["result"] = "fail"
            #print("fail item, target: %d" %(tgt))
            #print(pred[0])
        else:
            itemOut["result"] = "success"
        itemOut["expClassID"] = tgt 
        itemOut["expClassName"] = corpus.classNameList[tgt]

        itemOut["target"] = pred[0].tolist()
        allItemOut.append(itemOut)

    sucRate = 100 * (float(totalNum - failCnt)/float(totalNum))
    print("Total item: %d, Predict failed counter: %d, Total success rate: %.2f" %(testData.size(0), failCnt, sucRate )) 
    print("---------------------") 
    classSuccessRate = calculateSuccessRate(classCntList, classFailCntList)
    printCnt(corpus.classNameList, classCntList, classFailCntList, classSuccessRate, ['Name', 'Test Num', 'Fail Num', 'Success Rate'])
    print("---------------------") 

    jout["testNum"] = totalNum
    jout["classIDCntList"] = classCntList
    jout["classFailCntList"] = classFailCntList 
    jout["classNameTbl"] = corpus.classNameList 
    jout["successRate"] = sucRate 
    jout["successCnt"] = totalNum - failCnt 
    jout["failCnt"] = failCnt
    #jout["items"] = allItemOut 
    
    with open("testResult.json", "w") as outfile:
        json.dump(jout, outfile)


if __name__ == "__main__":
    runTestDataOneByOne()


    
