# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 11:38:48 2018

@author: Greydon
"""
import numpy as np
import pandas as pd
import scipy.io
import os
import re
import subprocess
import tables
import matplotlib.pyplot as plt
import json

combinatoDir = r'C:\Users\Greydon\Anaconda3\Lib\site-packages\combinato-master'
mainDir = r'D:\intraoperativeStudies\stopSignal\allData'
optimize = False

#%% Spike Extraction and Sorting

folders = []
for i in os.listdir(mainDir):
    if 'sub' in i:
        folders.append(i)

for isub in range(len(folders)): 
    subject = int(re.split('([0-9]+)', folders[isub])[1])
    dataFiles = os.listdir(os.path.join(mainDir,folders[isub],'spikeRaw'))
    outPutDir = os.path.join(mainDir,folders[isub],'spikeSorting')
    if not os.path.exists(outPutDir):
                os.makedirs(outPutDir)
                
    for idata in range(len(dataFiles)):
        data = {}
        data['data'] = pd.read_table(os.path.join(mainDir,folders[isub],'spikeRaw',dataFiles[idata]),  header=None).values
        filen = outPutDir + '\\' + dataFiles[idata][:-4] + '.mat'
        scipy.io.savemat(filen,data)
        
        changeDir = 'cd ' + outPutDir
        extract = 'python ' + combinatoDir + '/css-extract --matfile'
        cluster = 'python '+ combinatoDir + '/css-simple-clustering {} --datafile'
        
        filen = dataFiles[idata][:-4] + '.mat'
        newData = filen[:-4] + '/' + 'data_' + filen[:-4] + '.h5'
        
        #--- Extract
        command = extract + ' ' + filen
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE , shell=True, cwd=changeDir[3:])
        stdout = process.communicate()[0]
        
        if optimize == True:
            options = {'MaxClustersPerTemp': 7,
                   'RecursiveDepth': 2,
                   'MinInputSizeRecluster': 1000,
                   'MaxDistMatchGrouping': 1.6,
                   'MarkArtifactClasses': False,
                   'RecheckArtifacts': False}
            localOp = changeDir[3:] + '/' + filen[:-4] + '/local_options'
            np.save(localOp, options)
            os.rename(localOp + '.npy', localOp + '.py')
            
            commandNeg = cluster.format('--neg') + ' ' + newData + ' --label optimized'
            commandPos = cluster.format('') + ' ' + newData + ' --label optimized'
        else:
            commandNeg = cluster.format('--neg') + ' ' + newData
            commandPos = cluster.format('') + ' ' + newData
            
        #--- Sort Negative
        process = subprocess.Popen(commandNeg.split(), stdout=subprocess.PIPE , shell=True, cwd=changeDir[3:])
        stdout = process.communicate()[0]
        
        #--- Sort Positive
        process = subprocess.Popen(commandPos.split(), stdout=subprocess.PIPE , shell=True, cwd=changeDir[3:])
        stdout = process.communicate()[0]
        
        print("Done extracting/clustering file {} of {}: ".format(str(idata +1), str(len(dataFiles))),filen)

#%% Spike Result Import

detectionTypes = ['pos','neg']
removeArtifacts = True

folders = []
for i in os.listdir(mainDir):
    if 'sub' in i:
        folders.append(i)

      
for isub in range(2,2): 
    subject = int(re.split('([0-9]+)', folders[isub])[1])
    spikeTimesPos = []
    spikesPos = []
    spikesNeg = []
    spikeTimesNeg = []
    spikeResults = pd.DataFrame([])

    mat_files = [f for f in os.listdir(os.path.join(mainDir,folders[isub],'spikeSorting')) if f.endswith('.mat')]
    
    for ifile in range(len(mat_files)):
        filen = mat_files[ifile][:-4]
        channel = int(re.split('([0-9]+)',mat_files[ifile])[3])
        
        for idetect in range(len(detectionTypes)):
            checkClass = os.path.join(mainDir,folders[isub],'spikeSorting') + '/' + filen + '/sort_' + detectionTypes[idetect]  + '_simple/sort_cat.h5'
            spikesTemp = []
            spikeTimeTemp = []
            
            if os.path.isfile(checkClass):
                fid = tables.open_file(checkClass, 'r')
                allClass = np.unique(fid.get_node('/classes')[:])
                Types = fid.get_node('/artifacts')[:]
                
                if any(allClass!=0):
                    allClass = allClass[allClass>0]
                    
                    if removeArtifacts == True:
                        allClassFinal = []
                        for iclass in range(len(allClass)):
                            if Types[iclass,1] !=1:
                                allClassFinal.append(allClass[iclass])
                        fid.list_nodes
                        if len(allClassFinal) > 0:
                            classes = fid.get_node('/classes')[:]
                            matches = fid.get_node('/matches')[:]
                            fid.close()
                            h5File = os.path.join(mainDir,folders[isub],'spikeSorting') + '/' + filen + '/data_' + filen + '.h5'
                            fid = tables.open_file(h5File, 'r')
                            spk = fid.get_node('/' + detectionTypes[idetect] + '/spikes')[:, :]
                            spk = spk[(classes>0) & (matches>0),:]
                            time = fid.get_node('/' + detectionTypes[idetect] + '/times')[:]
                            spikesTemp.append(np.column_stack((classes[(classes>0) & (matches>0)], spk)))
                            spikeTimeTemp.append(time[(classes>0) & (matches>0)])
                            fid.close()
                        else:
                            spikesTemp = []
                            spikeTimeTemp = []
                    else:
                        classes = fid.get_node('/classes')[:]
#                        matches = fid.get_node('/matches')[:]
                        fid.close()
                        h5File = os.path.join(mainDir,folders[isub],'spikeSorting') + '/' + filen + '/data_' + filen + '.h5'
                        fid = tables.open_file(h5File, 'r')
                        spk = fid.get_node('/' + detectionTypes[idetect] + '/spikes')[:, :]
#                        spk = spk[(classes>0) & (matches>0),:]
                        spk = spk[(classes>0),:]
                        time = fid.get_node('/' + detectionTypes[idetect] + '/times')[:]
#                        spikesTemp.append(np.column_stack((classes[(classes>0) & (matches>0)], spk)))
#                        spikeTimeTemp.append(time[(classes>0) & (matches>0)])
                        spikesTemp.append(np.column_stack((classes[(classes>0)], spk)))
                        spikeTimeTemp.append(time[(classes>0)])
                        fid.close()
                else:
                    spikesTemp = []
                    spikeTimeTemp = []
                    fid.close()
            else:
                spikesTemp = []
                spikeTimeTemp = []
                
            if 'pos' in detectionTypes[idetect]:
                if len(spikesTemp) > 0:
                    spikesPos.append(spikesTemp[0])
                    spikeTimesPos.append(spikeTimeTemp[0])
                else:
                    spikesPos.append(spikesTemp)
                    spikeTimesPos.append(spikeTimeTemp)
            else:
                if len(spikesTemp) > 0:
                    spikesNeg.append(spikesTemp[0])
                    spikeTimesNeg.append(spikeTimeTemp[0])
                else:
                    spikesNeg.append(spikesTemp)
                    spikeTimesNeg.append(spikeTimeTemp)    
        temp = [{'subject': subject, 'channel': channel, 'PositiveSpikes': np.concatenate(spikesPos), 'PositiveTimes': np.concatenate(spikeTimesPos), 'NegativeSpikes': np.concatenate(spikesNeg), 'NegativeTimes': np.concatenate(spikeTimesNeg)}]
        spikeResults = pd.concat([spikeResults, pd.DataFrame(temp)], ignore_index=True)

    spikeResults = spikeResults[['subject', 'channel', 'PositiveSpikes', 'PositiveTimes', 'NegativeSpikes', 'NegativeTimes']]
    outputSpikeResult = os.path.join(mainDir,folders[isub],'spikeResults')
    if not os.path.exists(outputSpikeResult):
        os.makedirs(outputSpikeResult)
    spikeResults.to_pickle(outputSpikeResult + '/' + folders[isub] + '_spikeResults')

#%% Plotting
def plotLabels(ax,title, before, after, xZeroLabel):        
    if ((before+after) % 1000 != 0) and before !=0:
        xticks = np.linspace(0,before+after,12)
        labels = [str(x) for x in range(before*-1, after,125)]
        labels[labels.index('0')] = xZeroLabel
    else:
        xticks = np.linspace(0,before+after,11)
        labels = [str(x) for x in np.linspace(before*-1, after,11)]
        labels[labels.index('0.0')] = xZeroLabel
    
    ax.set_ylabel('Trial Number', fontsize=12, weight = 'bold')
    ax.set_title(title, fontsize=14, weight = 'bold')
    ax.set_xlim(0, before+after)
    ax.xaxis.set_ticks(xticks)
    ax.xaxis.set_ticklabels(labels)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    
def plotHistogram(spikeData, bin_size, axis):
    all_spikes = np.concatenate(tuple(spikeData['spikes']))
    duration = np.unique(spikeData['duration'])
    nbins = int(np.ceil(duration/bin_size))
    psth, bin_edges = np.histogram(all_spikes,bins=nbins,range=(0, nbins*bin_size))
    axis.plot(bin_edges[:-1], psth, drawstyle='steps-post')
    axis.set_ylabel('Spikes Per Second', fontsize=12, weight = 'bold')
    axis.set_xlabel('Time (ms)', fontsize=12, weight = 'bold', labelpad=20)
    for tick in axis.xaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
    for tick in axis.yaxis.get_major_ticks():
        tick.label1.set_fontweight('bold')
        
folders = []
for i in os.listdir(mainDir):
    if 'sub' in i:
        folders.append(i)
        
chanLabels = ['Cen','Ant','Pos','Med','Lat']

for isub in range(2,len(folders)): 
    subject = int(re.split('([0-9]+)', folders[isub])[1])
    dataFolder = 'D:/intraoperativeStudies/stopSignal/allData/' + folders[isub]
    outputDir = dataFolder + '/plots/spikePlots'
    if not os.path.exists(outputDir):
                os.makedirs(outputDir)
                
    spikeData = pd.read_pickle(dataFolder + '/spikeResults/' + folders[isub] + '_spikeResults')
    taskData = pd.read_excel(dataFolder + '/' + folders[isub] + '_task.xlsx')
    filen = dataFolder + '/' + folders[isub] + '_subjectInfo.json'
    with open(filen) as f:
        subjectInfo = json.load(f)
    Fs = subjectInfo['spike'][0]['Fs']
    
    # Define General Values
    go = taskData['trialTypeCode'] == 1
    stop = taskData['trialTypeCode'] == 2
    correct = taskData['accuracy1'] == 1
    incorrect = taskData['accuracy1'] == 0
    omission = taskData['accuracy1'] == 9
    RTsAbove = taskData['RT'] <1200
    RTsBelow = taskData['RT'] >200
    RTs = taskData['RT']
    SSD = taskData['SSD']
    SSDStopCorrect = SSD[stop & correct]
    SSDStopIncorrect = SSD[stop & incorrect]
    SSDall = np.concatenate([SSDStopCorrect,SSDStopIncorrect])
    blocks = pd.unique(taskData['blockNum'])
    beforeFixTimeTask = taskData['beforeFixation'].values
    afterFixTimeTask = taskData['afterFixation'].values
    responseTimeTask = taskData['afterResponse'].values
    ITITimeTask = taskData['ITICutoff'].values
    
    # Define task conditions
    goCorrect = taskData[go & correct & RTsBelow & RTsAbove]
    outlierGo = abs(goCorrect['RT'] - np.median(goCorrect['RT'])) < 2.5* np.std(goCorrect['RT'])
    goCorrect = goCorrect[outlierGo]
    goCorrectRTs = taskData['RT'][go & correct & RTsBelow & RTsAbove & outlierGo]
    meanGoRTs = np.mean(goCorrectRTs)
    goError = taskData[go & incorrect & RTsBelow & RTsAbove]
    goOmission = taskData[go & omission]
    stopCorrect = taskData[stop & correct]
    stopIncorrect = taskData[stop & incorrect & RTsBelow & RTsAbove]
    totGo = goCorrect.shape[0] + goError.shape[0]
    totStop = stopCorrect.shape[0] + stopIncorrect.shape[0]
    goAccuracy = goCorrect.shape[0]/totGo*100
    stopAccuracy = stopCorrect.shape[0]/totStop*100
    stopErrorAccurracy = stopIncorrect.shape[0]/totStop*100
    pErrorSignal = 100 - stopErrorAccurracy
    goRTPercentile = np.percentile(goCorrectRTs, pErrorSignal)
    SSRTmean = meanGoRTs - np.mean(SSDall)
    SSRTmedian = np.median(goCorrectRTs) - np.median(SSDall)
    SSRTquartile = goRTPercentile - np.mean(SSDall)

    goCorrectTrials = goCorrect['trialOverall'].values
    stopCorrectTrials = stopCorrect['trialOverall'].values
    stopIncorrectTrials = stopIncorrect['trialOverall'].values
    
    channels = np.unique(spikeData['channel'])
    mat_files = [f for f in os.listdir(os.path.join(mainDir,folders[isub],'spikeSorting')) if f.endswith('.mat')]
    
    for ichan in range(len(channels)):
        channel = channels[ichan]
        chanIdx = taskData.columns.get_loc(chanLabels[channel-1] + 'BeforeFix')
        dataMat = scipy.io.loadmat(os.path.join(mainDir,folders[isub],'spikeSorting',mat_files[ichan]))
        totalTime = int((len(dataMat['data'])/Fs)*1000)
        
        # Define Timestamps Values
        SSDTime = np.array([int(x) for x in taskData['SSD'].values])
        beepTimeTask = SSDTime + np.array(afterFixTimeTask)
        soundTimeSpike = taskData[chanLabels[channel-1] + 'SpikeHz']/(Fs/1000)
        soundIdx = np.where(~np.isnan(soundTimeSpike))[0]
        
        beforeFixTimestamp = []
        afterFixTimestamp = []
        responseTimestamp = []
        ITITimestamp = []
        
        if soundIdx[0]>0:
            initialSpikeIdx = soundTimeSpike[soundIdx[0]] - SSDTime[soundIdx[0]]
            initialTaskIdx = beepTimeTask[soundIdx[0]] - SSDTime[soundIdx[0]]
            taskIdx = soundIdx[0]
            
            for isound in range(soundIdx[0]+1):
                beforeFixDiff = beepTimeTask[taskIdx-isound]-beforeFixTimeTask[taskIdx-isound]
                afterFixDiff = beepTimeTask[taskIdx-isound]-afterFixTimeTask[taskIdx-isound]
                responseTimeDiff = beepTimeTask[taskIdx-isound]-responseTimeTask[taskIdx-isound]
                ITITimeDiff = beepTimeTask[taskIdx-isound]-ITITimeTask[taskIdx-isound]
                
                if not beforeFixTimestamp:
                    beforeFixTimestamp.append(int(soundTimeSpike[soundIdx[0]]) - beforeFixDiff)
                    afterFixTimestamp.append(int(soundTimeSpike[soundIdx[0]]) - afterFixDiff)
                    responseTimestamp.append(int(soundTimeSpike[soundIdx[0]]) - responseTimeDiff)
                    ITITimestamp.append(int(soundTimeSpike[soundIdx[0]]) - ITITimeDiff)
                    
                else:
                    diff = initialTaskIdx - beepTimeTask[taskIdx - isound]
                    afterFixTimestamp.append(initialSpikeIdx - diff)
                    beforeFixTimestamp.append(afterFixTimestamp[isound] - beforeFixDiff)
                    responseTimestamp.append(afterFixTimestamp[isound] - responseTimeDiff)
                    ITITimestamp.append(afterFixTimestamp[isound] - ITITimeDiff)
                    
            beforeFixTimestamp = beforeFixTimestamp[::-1]
            afterFixTimestamp = afterFixTimestamp[::-1]
            responseTimestamp = responseTimestamp[::-1]
            ITITimestamp = ITITimestamp[::-1]
            start = soundIdx[0] + 1
        else:
            start = 0
       
        for itime in range(start,len(beepTimeTask)):
            beforeFixDiff = beepTimeTask[itime]-beforeFixTimeTask[itime]
            afterFixDiff = beepTimeTask[itime]-afterFixTimeTask[itime]
            responseTimeDiff = beepTimeTask[itime]-responseTimeTask[itime]
            ITITimeDiff = beepTimeTask[itime]-ITITimeTask[itime]
            
            if itime in soundIdx:
                beforeFixTimestamp.append(soundTimeSpike[itime] - beforeFixDiff)
                afterFixTimestamp.append(soundTimeSpike[itime] - afterFixDiff)
                responseTimestamp.append(soundTimeSpike[itime] - responseTimeDiff)
                ITITimestamp.append(soundTimeSpike[itime] - ITITimeDiff)
                
            else:
                diff = afterFixTimeTask[itime] - afterFixTimeTask[itime-1]
                afterFixTimestamp.append(afterFixTimestamp[-1] + diff)
                beforeFixTimestamp.append(afterFixTimestamp[itime] - beforeFixDiff)
                responseTimestamp.append(afterFixTimestamp[itime] - responseTimeDiff)
                ITITimestamp.append(afterFixTimestamp[itime] - ITITimeDiff)
            
        beforeFixTimestamp = np.array(beforeFixTimestamp).astype(int)
        afterFixTimestamp = np.array(afterFixTimestamp).astype(int)
        responseTimestamp = np.array(responseTimestamp).astype(int)
        ITITimestamp = np.array(ITITimestamp).astype(int)
        last = spikeData.loc[spikeData['channel']==channel, 'PositiveTimes'].values[0]
        spikeTrainPos = np.zeros([1,totalTime])
        spikeTrainNeg = np.zeros([1,totalTime])
        spikeTrainPos[0,[int(x) for x in spikeData.loc[spikeData['channel']==channel, 'PositiveTimes'].values[0]]] = 1
        spikeTrainNeg[0,[int(x) for x in spikeData.loc[spikeData['channel']==channel, 'NegativeTimes'].values[0]]] = 1
        spikeTrainCombine = spikeTrainPos + spikeTrainNeg
        
        zeroPoints = {}
        zeroPoints['Entire Trial'] = [{'before': 0, 'after': 1500, 'timestamps': beforeFixTimestamp, 'xlabel': '0.0'}]
        zeroPoints['Stimulus'] = [{'before': 500, 'after': 1000, 'timestamps': afterFixTimestamp, 'xlabel': 'Stimulus'}]
        zeroPoints['Response'] = [{'before': 1000, 'after': 1000, 'timestamps': responseTimestamp, 'xlabel': 'Response'}]

        for iplot in range(len(zeroPoints)):
            before = zeroPoints[list(zeroPoints.keys())[iplot]][0]['before']
            after = zeroPoints[list(zeroPoints.keys())[iplot]][0]['after']
            timeStamps = zeroPoints[list(zeroPoints.keys())[iplot]][0]['timestamps']
           
            fig1, axs1 = plt.subplots(2,1, sharex=True)
            plt.subplots_adjust(hspace = .001)
            fig2, axs2 = plt.subplots(2,1, sharex=True)
            plt.subplots_adjust(hspace = .001)
            fig3, axs3 = plt.subplots(2,1, sharex=True)
            plt.subplots_adjust(hspace = .001)
            
            spikeTimesGoTrials = pd.DataFrame([])
            spikeTimesStopCorTrials = pd.DataFrame([])
            spikeTimesStopInTrials = pd.DataFrame([])
            trialGo = 0
            trialStopCor = 0
            trialStopIn = 0
            
            for trial in range(np.max(taskData['trialOverall'].values)):
                spikeTimesFin = spikeTrainCombine[0,timeStamps[trial]-before:timeStamps[trial]+after]
                spikeTime = np.where(spikeTimesFin>0)
                
                if (trial+1) in goCorrectTrials:
                    axs1[0].vlines(spikeTime,trialGo,trialGo+1)
                    trialGo +=1
                    
                    spikeTimesGoTrialsTemp = [{'cf': Fs, 'duration': before+after, 'spikes': spikeTime[0]}]
                    spikeTimesGoTrials = pd.concat([spikeTimesGoTrials, pd.DataFrame(spikeTimesGoTrialsTemp)], axis = 0, ignore_index=True)
                    
                elif (trial+1) in stopCorrectTrials:
                    axs2[0].vlines(spikeTime,trialStopCor,trialStopCor+1)
                    trialStopCor +=1
                    
                    spikeTimesStopCorTrialsTemp = [{'cf': Fs, 'duration': before+after, 'spikes': spikeTime[0]}]
                    spikeTimesStopCorTrials = pd.concat([spikeTimesStopCorTrials, pd.DataFrame(spikeTimesStopCorTrialsTemp)], axis = 0, ignore_index=True)
                    
                elif (trial+1) in stopIncorrectTrials:
                    axs3[0].vlines(spikeTime,trialStopIn,trialStopIn+1)
                    trialStopIn +=1
                    
                    spikeTimesStopInTrialsTemp = [{'cf': Fs, 'duration': before+after, 'spikes': spikeTime[0]}]
                    spikeTimesStopInTrials = pd.concat([spikeTimesStopInTrials, pd.DataFrame(spikeTimesStopInTrialsTemp)], axis = 0, ignore_index=True)
            
            plotLabels(axs1[0],'Go Correct Trials: ' + list(zeroPoints.keys())[iplot], before, after, xZeroLabel = zeroPoints[list(zeroPoints.keys())[iplot]][0]['xlabel'])
            plotLabels(axs2[0],'Stop Correct Trials: ' + list(zeroPoints.keys())[iplot], before, after, xZeroLabel = zeroPoints[list(zeroPoints.keys())[iplot]][0]['xlabel'])
            plotLabels(axs3[0],'Stop Incorrect Trials: ' + list(zeroPoints.keys())[iplot], before, after, xZeroLabel = zeroPoints[list(zeroPoints.keys())[iplot]][0]['xlabel'])
            
            plotHistogram(spikeTimesGoTrials, 15, axs1[1])
            plotHistogram(spikeTimesStopCorTrials, 15, axs2[1])
            plotHistogram(spikeTimesStopInTrials, 15, axs3[1])
            
            fileName = folders[isub] + '_chan-' + str(channel) + '_type-' + 'GoCorrect' + '_zeropoint-' + list(zeroPoints.keys())[iplot] + '.png'
            fig1.set_size_inches(12, 8)
            filepath = outputDir + '/'+ fileName
            fig1.savefig(filepath, dpi=100)   # save the figure to file
            
            fileName = folders[isub] + '_chan-' + str(channel) + '_type-' + 'StopCorrect' + '_zeropoint-' + list(zeroPoints.keys())[iplot] + '.png'
            fig2.set_size_inches(12, 8)
            filepath = outputDir + '/'+ fileName
            fig2.savefig(filepath, dpi=100)   # save the figure to file
            
            fileName = folders[isub] + '_chan-' + str(channel) + '_type-' + 'StopIncorrect' + '_zeropoint-' + list(zeroPoints.keys())[iplot] + '.png'
            fig3.set_size_inches(12, 8)
            filepath = outputDir + '/'+ fileName
            fig3.savefig(filepath, dpi=100)   # save the figure to file
            
            plt.close('all')

        
