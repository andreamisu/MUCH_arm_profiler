import sys, logging
from rich.table import Table
from rich.console import Console
from datetime import datetime
import numpy
import math
import subprocess
import argparse
import itertools
from utils import *
from ctypes import *
import os
import errno
from matplotlib import mlab
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, multivariate_normal, norm
from mpl_toolkits import mplot3d
import pickle 
from sklearn import linear_model
import pandas
import seaborn as sns
import time
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor

BENCHMARK_STATISTICS_FILE = './benchmark_statistics.dump'
BENCHMARK_PMU_FILE = './benchmark_pmus.dump'
PMU_GROUPED_HI_FILE = './pmu_grouped.dump'
PERF_COMMAND = "./profiling"
PERF_LIST_FILENAME = "./pmu_lists/perf.armA53.list"
PERF_REPORT_FILENAME = "benchmarks.out"
RUN_COUNTER = 0
PMU_STEPS = 5
RAW_PMU = {} #key: pmu name / value: hex raw pmu
RUNS_FAILED = []
EVENTS_THRESHOLD = 20000
MUCH_BENCH_PMUS = []
EXPERIMENTS_LIST = []
ALLOCABLE_PMUS = 6
MUCH_RUNS = 50 # 30 is the minimum suggestabele from the paper in order to use Central Limit Theory. Higher the number, More precise the measurement of correlations.
OPTIMIZZATION_STEPS = 10

PMU_GROUPED_HI = {}
PMU_STATISTICS = {} #key: pmu name
PMU_MAPPED = {}


EXPERIMENTS_RESULTS_TABLE = []  #main obejct in which store results from experiments
# {
#     index: index of experiments,
#     pmus: [pmu names] of experiments,
#     data: [] array with every experiment data inside
# }

EFFICIENT_PMUS_ALLOCATION = 5 # EPA => # MUCH_BENCH_PMUS - (EPA + (EPA-1 * EPA-1)) =(or near) 0      so in the case of 30 allocable valid MUCH PMUs, EPA is 6 as 6 + 5 * 5 = 31     this is mostly helpful for correlation runs

MUCH_EXECUTED_ITERATION = {} #each index i refers to MUCH_BENCH_PMUS[i], and basically contains which PMUs has been already checked with the given event monitor.


#mean square error of two matrix
def mse(actual, predicted):
    return numpy.square(numpy.subtract(actual, predicted)).mean()


def pmu_allocation(MUCH_BENCH_PMUS, max):
    numPMU = len(MUCH_BENCH_PMUS)
    i = 0
    while (math.pow(i, 2) - i + 1 < numPMU):
        i += 1
    if i == 0:
        print('error calculating pmu_allocation. Shutting down...')
        exit()
    elif math.pow(i, 2) - i + 1 == numPMU:
        return MUCH_BENCH_PMUS
    else:
        count = 0
        while(math.pow(i, 2) - i + 1 != len(MUCH_BENCH_PMUS)):
            MUCH_BENCH_PMUS.append('fakePMU%d' % count)
            count += 1
        return MUCH_BENCH_PMUS


def writeLogsFile(out, pmuSelected):
    global table, RUN_COUNTER, MUCH_BENCH_PMUS, console
    # write to benchmark output log
    fileObject = open(PERF_REPORT_FILENAME, 'a')
    fileObject.write(out)
    fileObject.close() 

    # write to aux file for table population 
    fileObject = open("aux.temp", 'w')
    fileObject.write(out)
    fileObject.close()
    
    pmuList = pmuSelected.split(",")

    fileObject = open("aux.temp", 'r')
    fileLines = fileObject.readlines()

    for line in fileLines:
        if any(pmu in line for pmu in pmuList):
            pmuData = list(filter(None, line.split(" ")))
            pmuData[2] = "(100.00%)" if pmuData[2].replace("\n","") == "" else pmuData[2].replace("\n","")
            if "supported" not in str(pmuData[1]):
                table.add_row( pmuData[1], str(pmuData[0]), pmuData[2], str(RUN_COUNTER))
                if int(pmuData[0].replace(",","")) > EVENTS_THRESHOLD:
                    MUCH_BENCH_PMUS.append(pmuData[1])
            else:
                table.add_row( pmuData[2], "<Not Supported>", "", str(RUN_COUNTER))
    fileObject.close()


def initLogs():
    global table
    table = Table(title="PMUs Evaluation")
    table.add_column("PMU Name", style="cyan", no_wrap=True)
    table.add_column("# Events", justify="right",style="magenta")
    table.add_column("# Run", justify="right", style="green")

    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    fileObject = open(PERF_REPORT_FILENAME, 'w')
    fileObject.write("--------------")
    fileObject.write("START BENCHING: " + dt_string)
    fileObject.write("--------------")
    fileObject.close()


def main():
    global table, RUN_COUNTER, args, MUCH_BENCH_PMUS, MUCH_EXECUTED_ITERATION, EXPERIMENTS_RESULTS_TABLE, console
    
    if args.sudo:
        if not getSudoPermissions():
            print("User not getting sudo permissions, exit..")
            sys.exit(-1)

    
    initLogs()
    console = Console()
    perfList = []

    with console.status("working around...", spinner="monkey"):
        with open(PERF_LIST_FILENAME) as f:
            for line in f.readlines():
                if "#" not in line: #filter out comments
                    console.print(line)
                    pmu = line.replace("\n","").split("/")[1]
                    perfList.append(pmu)
                    RAW_PMU[pmu] = line.replace("\n","").split("/")[2]


    console.print("numbers of fetchable PMU events: %d" % (len(perfList)))
    console.print(",      ".join(perfList), style="green")
    console.print("numbers of PMU events each run: " + str(PMU_STEPS))
    
    for x in range(PMU_STEPS, len(perfList), PMU_STEPS):
        pmuSelected = ''
        for idx, val in enumerate(perfList[x-PMU_STEPS:x]):
            pmuSelected += RAW_PMU[val] if idx == 0 else "/"+RAW_PMU[val]
        cmdBench = ["sudo", PERF_COMMAND if args.sudo else PERF_COMMAND , pmuSelected]
        with console.status("doing benchmarks on {} ...".format(pmuSelected.replace(",",", "))):
            results = subprocess.run(cmdBench, text=True, capture_output=True)
            if(results.returncode == 0):
                logging.debug(results.stdout)
                for idx, val in enumerate(perfList[x-PMU_STEPS:x]):
                    events = int(results.stdout.split('/')[idx])
                    table.add_row(val, str(events), str(RUN_COUNTER))
                    if events > EVENTS_THRESHOLD:
                        MUCH_BENCH_PMUS.append(val)
            else:
                RUNS_FAILED.append(RUN_COUNTER)
        for pmuEvaluated in perfList[x-PMU_STEPS:x]:
            console.print(pmuEvaluated + (" :heavy_check_mark:" if  results.returncode == 0 else " :warning:"))
        RUN_COUNTER += 1

    console.print(table)
    console.print("benchmarks done!", style="blink")
    if(len(RUNS_FAILED) > 0):
        console.print("Failed runs: " + str(RUNS_FAILED), style="red")
    console.print("check report file in " + PERF_REPORT_FILENAME)
    console.print("number of PMUs available for MUCH evaluation: %d" % (len(MUCH_BENCH_PMUS)))

    # Algorithm for best allocation on given PMUs
    MUCH_BENCH_PMUS = pmu_allocation(MUCH_BENCH_PMUS, ALLOCABLE_PMUS)
    logging.debug("number of PMUs available for MUCH evaluation: %d" % (len(MUCH_BENCH_PMUS)))
    logging.debug(MUCH_BENCH_PMUS)

    for pivotPMU in MUCH_BENCH_PMUS:
        try:
            len(MUCH_EXECUTED_ITERATION[pivotPMU])
        except KeyError:
            MUCH_EXECUTED_ITERATION[pivotPMU] = []
        while len(set(MUCH_EXECUTED_ITERATION[pivotPMU])) < len(MUCH_BENCH_PMUS):
            chosenMuchPmus = [pivotPMU]
            for secondaryPMU in list(filter(lambda elm: elm not in MUCH_EXECUTED_ITERATION[pivotPMU], MUCH_BENCH_PMUS)):
                if secondaryPMU == pivotPMU:
                    continue
                if checkComplementaryPMUIterations(chosenMuchPmus, secondaryPMU):
                    chosenMuchPmus.append(secondaryPMU)
                    if len(chosenMuchPmus) == EFFICIENT_PMUS_ALLOCATION:
                        break #reached max allocable pmus

            for x in chosenMuchPmus:
                try:
                    len(MUCH_EXECUTED_ITERATION[x])
                except KeyError:
                    MUCH_EXECUTED_ITERATION[x] = []
                MUCH_EXECUTED_ITERATION[x].extend(chosenMuchPmus)
            EXPERIMENTS_LIST.append(chosenMuchPmus)

    for idx,elm in enumerate(EXPERIMENTS_LIST):
        k = [y for y in elm if "fakePMU" not in y]
        EXPERIMENTS_LIST[idx] = k

    MUCH_BENCH_PMUS = [y for y in MUCH_BENCH_PMUS if "fakePMU" not in y]
    logging.debug("experiments: \n%s" % (str(EXPERIMENTS_LIST)))

    # starts experiments
    for index in range(0,len(EXPERIMENTS_LIST)):
        initalizeExperimentObject(EXPERIMENTS_LIST[index])
        pmuSelected = ''
        for idx, val in enumerate(EXPERIMENTS_LIST[index]):
            pmuSelected += RAW_PMU[val] if idx == 0 else "/"+RAW_PMU[val]
        # console.print(pmuSelected)
        cmdBench = ["sudo", PERF_COMMAND if args.sudo else PERF_COMMAND , pmuSelected]
        for i in range(0, MUCH_RUNS):
            with console.status("Benchmark for experiment # {} \n {} \n {} / {} runs".format(index+1, ",".join(EXPERIMENTS_LIST[index]), i, MUCH_RUNS)):
                results = subprocess.run(cmdBench, text=True, capture_output=True)
                if(results.returncode == 0):
                    collectMUCHValues(results.stdout, index, EXPERIMENTS_LIST[index])
                else:
                    console.print("unexpected error")
                    return -1

    if(args.write):
        data = {
            'EXPERIMENTS_LIST': EXPERIMENTS_LIST,
            'PMU_SUPPORTED' : MUCH_BENCH_PMUS,
            'EXPERIMENTS_RESULTS_TABLE': EXPERIMENTS_RESULTS_TABLE
        }
        with open(args.write, 'wb') as f:
            pickle.dump(data, f)
        console.print("Experiments done! data is exported at: %s" % args.write)
    drawingData()

    

def drawingData():
    global PMU_STATISTICS, MUCH_BENCH_PMUS, console, EXPERIMENTS_RESULTS_TABLE

    for h in MUCH_BENCH_PMUS:
        PMU_GROUPED_HI[h] = []
        for index in range(0,len(EXPERIMENTS_LIST)):
            if h not in EXPERIMENTS_LIST[index]:
                # pmu (h) not in experiment
                continue
            logging.debug("PMU %s is in experiment: %s" % (h, str(EXPERIMENTS_LIST[index])))
            for data in EXPERIMENTS_RESULTS_TABLE[index]["data"]:
                for subexp in data:
                    logging.debug("subexp[pmu]: %s" % (subexp["pmu"]))
                    if subexp["pmu"] == h:
                        PMU_GROUPED_HI[h].append(int(subexp["events"].replace(",","")))
                        logging.debug("events: %s" % (subexp["events"]))
        console.print('{}={}'.format(h,PMU_GROUPED_HI[h]))
        PMU_MAPPED[h] = {}
        num = len(PMU_GROUPED_HI[h])
        steps,dist = numpy.linspace(0,1,num+1,endpoint=False, dtype=numpy.float64, retstep=True)
        steps = steps.tolist()
        steps.remove(0)
        meanSteps = numpy.mean(steps)
        mvgdSampled = list(map(lambda elm: norm.ppf(elm, loc=0, scale=1), steps)) 

        pmu1_indexes_sort = numpy.argsort(PMU_GROUPED_HI[h])
        pmu1_argsort = numpy.zeros(len(PMU_GROUPED_HI[h]))
        for idx in range(0,len(pmu1_argsort)):
            pmu1_argsort[pmu1_indexes_sort[idx]] = mvgdSampled[idx]
        
        # console.print('PMU_GROUPED_HI[h]: {}'.format(PMU_GROUPED_HI[h]))
        # console.print('pmu1_indexes_sort: {}'.format(pmu1_indexes_sort))
        # console.print('mvgdSampled: {}'.format(mvgdSampled))
        # console.print('pmu1_argsort: {}'.format(pmu1_indexes_sort))

        for idx,elm in enumerate(PMU_GROUPED_HI[h]):
            PMU_MAPPED[h][elm] = pmu1_argsort[idx]
        # console.print('PMU_MAPPED[h]: {}'.format(PMU_MAPPED[h]))


        PMU_STATISTICS[h] = {
            "u": numpy.mean(PMU_GROUPED_HI[h], dtype = numpy.float64),
            "o": numpy.std(PMU_GROUPED_HI[h], dtype = numpy.float64), #standard deviation of hi subexperiment in PMU_GROUPED_HI[h] on u mean
            "o2": numpy.var(PMU_GROUPED_HI[h], dtype = numpy.float64), #variance of hi subexperiment in PMU_GROUPED_HI[h] on u mean
            "o2_mapped": 0,
            "o_mapped": 0,
            "u_mapped": 0,
            "o_pair": [],
            "p": [],
            "mvgdP": []     
        }

        # console.print("++++++++++++++++++++++++")
        # console.print("PMU : %s" % (h))
        # console.print("PMU_GROUPED_HI : %s" % (PMU_GROUPED_HI[h]))
        # console.print("PMU expectedValue (u) value: %s" % str(PMU_STATISTICS[h]["u"]))
        # console.print("PMU standard deviation (o) value: %s" % str(PMU_STATISTICS[h]["o"]))
        # console.print("PMU variance (o2) value: %s" % str(PMU_STATISTICS[h]["o2"]))

    # empirical correlation between 2 hems  ρˆij 
    counter = 0
    for pmu_couple in itertools.combinations(MUCH_BENCH_PMUS, 2):
        for index in range(0,len(EXPERIMENTS_LIST)):
            #experiment in which both are present
            if pmu_couple[0] in EXPERIMENTS_LIST[index] and pmu_couple[1] in EXPERIMENTS_LIST[index]:

                logging.debug("++++++++++++++++++++++++")
                logging.debug("correlation exp: %s + %s" % (pmu_couple[0], pmu_couple[1]))
                logging.debug("exp: %s" % str(EXPERIMENTS_LIST[index]))
                pmu1 = []
                pmu2 = []
                p = 0
                for data in EXPERIMENTS_RESULTS_TABLE[index]["data"]:
                    #console.print("data: %s" % data)
                    x = -1
                    y = -1
                    for subexp in data:
                        #console.print("subexp: %s" % subexp)
                        if subexp["pmu"] == pmu_couple[0]:
                            x = int(subexp["events"].replace(",",""))
                        if subexp["pmu"] == pmu_couple[1]:
                            y = int(subexp["events"].replace(",",""))
                    if x == -1 or y == -1:
                        console.print("Error: didn't find empirical correlation")
                        return
                    pmu1.append(x)
                    pmu2.append(y)
                # pearson correlation
                p, pvalue = pearsonr(pmu1, pmu2)
                logging.debug('{} + {} correlation: {}'.format(pmu_couple[0], pmu_couple[1],p))
                console.print('pmu1: {}'.format(pmu1))
                console.print('pmu2: {}'.format(pmu2))

                pmu1_argsort = []
                pmu2_argsort = []
                for elm in pmu1:
                    pmu1_argsort.append(PMU_MAPPED[pmu_couple[0]][elm])

                for elm in pmu2:
                    pmu2_argsort.append(PMU_MAPPED[pmu_couple[1]][elm])

                logging.debug('pmu_couple0: {}'.format(pmu_couple[0]))
                logging.debug('pmu_couple1: {}'.format(pmu_couple[1]))
                logging.debug('pmu1: {}'.format(pmu1))
                logging.debug('pmu2: {}'.format(pmu2))
                logging.debug('pmu1_argsort: {}'.format(pmu1_argsort))
                logging.debug('pmu2_argsort: {}'.format(pmu2_argsort))
                

                p_mapped, pvalue_mapped = pearsonr(pmu1_argsort, pmu2_argsort)
                console.print('%s + %s correlation: %f' % (pmu_couple[0], pmu_couple[1],p_mapped))


                ## tenere queste variance e std per creatione correlation matrix MVGD
                pmu1_mapped_obj = {
                    "u": numpy.mean(pmu1_argsort, dtype = numpy.float64),
                    "o": numpy.std(pmu1_argsort, dtype = numpy.float64), #standard deviation of hi subexperiment in PMU_GROUPED_HI[h] on u mean
                    "o2": numpy.var(pmu1_argsort, dtype = numpy.float64), #variance of hi subexperiment in PMU_GROUPED_HI[h] on u mean   
                }

                pmu2_mapped_obj = {
                    "u": numpy.mean(pmu2_argsort, dtype = numpy.float64),
                    "o": numpy.std(pmu2_argsort, dtype = numpy.float64), #standard deviation of hi subexperiment in PMU_GROUPED_HI[h] on u mean
                    "o2": numpy.var(pmu2_argsort, dtype = numpy.float64), #variance of hi subexperiment in PMU_GROUPED_HI[h] on u mean   
                }

                logging.debug('pmu1_mapped_obj: {}'.format(pmu1_mapped_obj))
                logging.debug('pmu2_mapped_obj: {}'.format(pmu2_mapped_obj))                

                PMU_STATISTICS[pmu_couple[0]]["o_mapped"] = pmu1_mapped_obj["o"]
                PMU_STATISTICS[pmu_couple[0]]["o2_mapped"] = pmu1_mapped_obj["o2"]
                PMU_STATISTICS[pmu_couple[0]]["u_mapped"] = pmu1_mapped_obj["u"]

                PMU_STATISTICS[pmu_couple[1]]["o_mapped"] = pmu2_mapped_obj["o"]
                PMU_STATISTICS[pmu_couple[1]]["o2_mapped"] = pmu2_mapped_obj["o2"]
                PMU_STATISTICS[pmu_couple[1]]["u_mapped"] = pmu2_mapped_obj["u"]
                

                PMU_STATISTICS[pmu_couple[0]]["o_pair"].append({
                    "pair" : pmu_couple[1],
                    "val": p * PMU_STATISTICS[pmu_couple[0]]["o"] * PMU_STATISTICS[pmu_couple[1]]["o"],
                    "val_mapped": p_mapped * pmu1_mapped_obj["o"] * pmu2_mapped_obj["o"],
                    'obj_mapped': pmu1_mapped_obj
                })


                PMU_STATISTICS[pmu_couple[1]]["o_pair"].append({
                    "pair" : pmu_couple[0],
                    "val": p * PMU_STATISTICS[pmu_couple[0]]["o"] * PMU_STATISTICS[pmu_couple[1]]["o"],
                    "val_mapped": p_mapped * pmu1_mapped_obj["o"] * pmu2_mapped_obj["o"],
                    'obj_mapped': pmu2_mapped_obj
                })

                PMU_STATISTICS[pmu_couple[0]]["p"].append({
                    "pair" : pmu_couple[1],
                    "val": p,
                    'val_mapped': p_mapped
                })

                PMU_STATISTICS[pmu_couple[1]]["p"].append({
                    "pair" : pmu_couple[0],
                    "val": p,
                    'val_mapped': p_mapped
                })

                logging.debug("pmu %s o: %f" %(pmu_couple[0], PMU_STATISTICS[pmu_couple[0]]["o"]))
                logging.debug("pmu %s o: %f" %(pmu_couple[1], PMU_STATISTICS[pmu_couple[1]]["o"]))
                logging.debug("%s + %s:\no_pair > %s \np: %s" % (pmu_couple[0], pmu_couple[1], str(p * PMU_STATISTICS[pmu_couple[0]]["o"] * PMU_STATISTICS[pmu_couple[1]]["o"]), str(p)))

    #http://users.stat.umn.edu/~helwig/notes/datamat-Notes.pdf
    #Correlation matrix S
    correlationMap = []
    for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
        console.print("pmu1: %s" % pmu1)
        correlationLine = []
        for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
            console.print("pmu2: %s" % pmu2)
            if pmu1==pmu2:
                console.print("same")
                correlationLine.append(float(1)) #correlation between same values is 1
            else:
                for corr in PMU_STATISTICS[pmu2]['p']:
                    if corr['pair'] == pmu1:
                        correlationLine.append(corr['val'])
        correlationMap.append(correlationLine)
    console.print("++++++++++++++++++++++++++++++++++")
    console.print(correlationMap)
    correlationMatrix = numpy.array(correlationMap, dtype=numpy.float64)
    console.print("correlation matrix: {}".format(correlationMatrix))
    console.print("++++++++++++++++++++++++++++++++++")

    #Covariance matrix Ʃˆ
    mean_array = []
    for pmu in MUCH_BENCH_PMUS:
        mean_array.append(PMU_STATISTICS[pmu]['u'])

    covarianceMap = []
    for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
        console.print("pmu1: %s" % pmu1)
        covarianceLine = []
        for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
            console.print("pmu2: %s" % pmu2)
            if pmu1==pmu2:
                covarianceLine.append(PMU_STATISTICS[pmu2]['o2'])
            else:
                for corr in PMU_STATISTICS[pmu1]['o_pair']:
                    if corr['pair'] == pmu2:
                        covarianceLine.append(corr['val'])
                        continue
        covarianceMap.append(covarianceLine)
        # console.print("PMU %s : %s" % (pmu1, str(covarianceLine)))
        # PMU_STATISTICS[pmu1]['mvgd'] = multivariate_normal(mean=mean_array, cov=covarianceLine)
    console.print("++++++++++++++++++++++++++++++++++")
    console.print("covariance matrix:")
    covarianceMatrix = numpy.array(covarianceMap, dtype = numpy.float64)
    console.print(covarianceMatrix)

    #multivariate normal MVGD  𝑋 ∼ N𝑛ℎ(𝜇,ˆ Σˆ)
    # mean_array = []
    # for pmu in MUCH_BENCH_PMUS:
    #     mean_array.append(PMU_STATISTICS[pmu]['u'])
    # console.print("mean array: %s" % mean_array)
    # # mvgd = multivariate_normal.pdf(mean=mean_array, cov=covarianceMatrix)
    # mvgd = multivariate_normal(mean=mean_array, cov=covarianceMatrix, allow_singular=True)
    # console.print(mvgd)


    #Application of copula theory
    for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
        num = len(PMU_GROUPED_HI[pmu1])
        steps,dist = numpy.linspace(0,1,num+1,endpoint=False, dtype=numpy.float64, retstep=True)
        steps = steps.tolist()
        steps.remove(0)
        meanSteps = numpy.mean(steps)
        console.print("num is %d" % num)
        console.print("steps are %s" % str(steps))

        # A uniform sample can be transformed into a
        # Gaussian sample by applying the inverse function 
        # of the cumulative distribution function of a standard Gaussian distribution, Φ

        # Percent point function (inverse of cdf — percentiles).
        mvgdSampled = list(map(lambda elm: norm.ppf(elm, loc=0, scale=1), steps)) 
        console.print(mvgdSampled)
        PMU_STATISTICS[pmu1]['ppf'] = {
                "val" : mvgdSampled,
                "u": numpy.mean(mvgdSampled),
                "o": numpy.std(mvgdSampled), #standard deviation
                "o2": numpy.var(mvgdSampled), #variance
                "p": [],
                "o_pair": []
        }

    for pmu_couple in itertools.combinations(MUCH_BENCH_PMUS, 2):      
        p, _ = pearsonr(PMU_STATISTICS[pmu_couple[0]]['ppf']['val'], PMU_STATISTICS[pmu_couple[1]]['ppf']['val'])
        console.print("ppf0 -> " + str(PMU_STATISTICS[pmu_couple[0]]['ppf']))
        console.print("ppf1 -> " + str(PMU_STATISTICS[pmu_couple[1]]['ppf']))
        console.print("covariance : {}".format(numpy.cov([PMU_STATISTICS[pmu_couple[0]]['ppf']['val'],PMU_STATISTICS[pmu_couple[0]]['ppf']['val']])))
        console.print("p : {}".format(p))
        PMU_STATISTICS[pmu_couple[0]]['ppf']["p"].append({
            "pair" : pmu_couple[1],
            "val": p
        })

        PMU_STATISTICS[pmu_couple[1]]['ppf']["p"].append({
            "pair" : pmu_couple[0],
            "val": p
        })
        PMU_STATISTICS[pmu_couple[0]]['ppf']["o_pair"].append({
            "pair" : pmu_couple[1],
            "val": p * PMU_STATISTICS[pmu_couple[0]]['ppf']["o"] * PMU_STATISTICS[pmu_couple[1]]['ppf']["o"]
        })
        PMU_STATISTICS[pmu_couple[1]]['ppf']["o_pair"].append({
            "pair" : pmu_couple[0],
            "val": p * PMU_STATISTICS[pmu_couple[0]]['ppf']["o"] * PMU_STATISTICS[pmu_couple[1]]['ppf']["o"]
        })
        
    correlationMap = []
    for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
        console.print("pmu1: %s" % pmu1)
        correlationLine = []
        for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
            console.print("pmu2: %s" % pmu2)
            if pmu1==pmu2:
                console.print("same")
                correlationLine.append(float(1)) #correlation between same values is 1
            else:
                for corr in PMU_STATISTICS[pmu2]['p']:
                    if corr['pair'] == pmu1:
                        correlationLine.append(corr['val_mapped'])
        correlationMap.append(correlationLine)
    console.print("++++++++++++++++++++++++++++++++++")
    console.print("correlation matrix:")
    console.print(correlationMap)
    gaussianCorrelationMatrix = numpy.array(correlationMap, dtype=numpy.float64)
    console.print("gaussianCorrelationMatrix: {}".format(gaussianCorrelationMatrix))

    #Covariance matrix Ʃˆ0
    gaussianCovarianceMap = []
    for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
        console.print("pmu1: %s" % pmu1)
        covarianceLine = []
        for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
            console.print("pmu2: %s" % pmu2)
            if pmu1==pmu2:
                covarianceLine.append(PMU_STATISTICS[pmu1]['o2_mapped'])
            else:
                for corr in PMU_STATISTICS[pmu1]['o_pair']:
                    if corr['pair'] == pmu2:
                        console.print("++++++++++++++++++++++++++++++++++")
                        console.print("corr['pair']: {}".format(corr['pair']))
                        console.print("corr['val']: {}".format(corr['val_mapped']))
                        covarianceLine.append(corr['val_mapped'])
                        continue
        gaussianCovarianceMap.append(covarianceLine)
    console.print("++++++++++++++++++++++++++++++++++")
    console.print("covariance matrix:")

    gaussianCovarianceMatrix = numpy.matrix(gaussianCovarianceMap, dtype = numpy.float64)
    console.print("gaussianCovarianceMatrix: {}".format(gaussianCovarianceMatrix))
    console.print("covarianceMatrix: {}".format(covarianceMatrix))

    console.print("correlationMatrix: {}".format(correlationMatrix))
    console.print("gaussianCorrelationMatrix: {}".format(gaussianCorrelationMatrix))
         
    #multivariate normal MVGD 𝑋 ∼ N𝑛ℎ(0, Σˆ0)
    new_correlation = []
    for iteration in range(0,OPTIMIZZATION_STEPS):
        correlationoject = {}
        for x in MUCH_BENCH_PMUS:
            correlationoject[x] = []

        
        vector_means = numpy.zeros(len(MUCH_BENCH_PMUS))
        sampMVGD = numpy.random.multivariate_normal(numpy.zeros(len(MUCH_BENCH_PMUS)),gaussianCovarianceMatrix,len(PMU_GROUPED_HI[pmu1]))
        HEMvector = []
        for idx,pmu in enumerate(MUCH_BENCH_PMUS):
            samp_array = sampMVGD[:,idx]
            index_samples = numpy.argsort(samp_array)       
            descOrdHEM = numpy.sort(PMU_GROUPED_HI[pmu])  
            regrouped = numpy.zeros(len(index_samples))
            for idx, elm in enumerate(index_samples):
                regrouped[elm] = descOrdHEM[idx]
            HEMvector.append(regrouped)
        HEMVectorMatrix = numpy.array(HEMvector, dtype=numpy.float64)

        tableP = Table(title="Correlation Evaluation")
        tableP.add_column("PMU 1", style="cyan", no_wrap=True)
        tableP.add_column("PMU 2",style="cyan", no_wrap=True)
        tableP.add_column("Empirical Correlation", justify="right", style="magenta")
        tableP.add_column("MVGD Correlation", justify="right", style="magenta")
        tableP.add_column("Correlation Delta", justify="right", style="green")

        for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
            for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
                console.print("len MUCH_BENCH: {}".format(MUCH_BENCH_PMUS))
                console.print("index1: {}".format(index1))
                console.print("index2: {}".format(index2))
                if index2 <= index1:
                    continue #same PMU
                pmu1Values = HEMVectorMatrix[index1,:]
                pmu2Values = HEMVectorMatrix[index2,:]
                mvgdP, _ = pearsonr(pmu1Values, pmu2Values)
                console.print(mvgdP)
                for pair in PMU_STATISTICS[pmu1]["p"]:
                    if(pair["pair"] == pmu2):
                        empiricalP = pair["val"]
                        deltaP = abs(mvgdP - empiricalP)
                        console.print('mvgdP: {}'.format(mvgdP))
                        console.print('empiricalP: {}'.format(empiricalP))
                        # PMU_STATISTICS[pmu1]['mvgdP'].append({
                        #     'pair': pmu2,
                        #     'value_p': mvgdP,
                        #     'delta_p': deltaP

                        # })
                        # PMU_STATISTICS[pmu2]['mvgdP'].append({
                        #     'pair': pmu1,
                        #     'value_p': mvgdP,
                        #     'delta_p': deltaP
                        # })
                        correlationoject[pmu1].append({
                            'pair': pmu2,
                            'value_p': mvgdP,
                            'delta_p': deltaP
                        })
                        correlationoject[pmu2].append({
                            'pair': pmu1,
                            'value_p': mvgdP,
                            'delta_p': deltaP
                        })
                        tableP.add_row(pmu1, pmu2, str(empiricalP), str(mvgdP), str(deltaP))
        console.print(tableP)

        # optimization step
        correlationMap = []
        for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
            console.print("pmu1: %s" % pmu1)
            correlationLine = []
            for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
                console.print("pmu2: %s" % pmu2)
                if pmu1==pmu2:
                    console.print("same")
                    correlationLine.append(float(1)) #correlation between same values is 1
                else:
                    # for corr in PMU_STATISTICS[pmu1]['mvgdP']:
                    for corr in correlationoject[pmu1]:
                        if corr['pair'] == pmu2:
                            correlationLine.append(corr['value_p'])
                    
            correlationMap.append(correlationLine)
        console.print("++++++++++++++++++++++++++++++++++")
        console.print("correlation matrix:")
        mvgdCorrelationMatrix = numpy.array(correlationMap, dtype=numpy.float64)
        console.print(mvgdCorrelationMatrix)

        #https://stats.stackexchange.com/questions/28461/how-to-denote-element-wise-difference-of-two-matrices
        #For calculating the MSE, you have to subtract every element of matrix 2 from every element of matrix 1

        mse = 0
        console.print("correlationMatrix: {}".format(correlationMatrix))
        console.print("mvgdCorrelationMatrix: {}".format(mvgdCorrelationMatrix))
        correlationDeltaMatrix = []
        for index_i in range(0,len(MUCH_BENCH_PMUS)):
            correlationDeltaArray = []
            for index_j in range(0,len(MUCH_BENCH_PMUS)):
                console.print("correlationMatrix[{}][{}]: {}".format(index_i, index_j, correlationMatrix[index_i][index_j]))
                console.print("correlationMatrix[{}][{}]: {}".format(index_i, index_j, mvgdCorrelationMatrix[index_i][index_j]))
                correlationDeltaArray.append(abs(correlationMatrix[index_i][index_j] - mvgdCorrelationMatrix[index_i][index_j]))
                mse += math.pow(correlationMatrix[index_i][index_j] - mvgdCorrelationMatrix[index_i][index_j], 2)
            correlationDeltaMatrix.append(correlationDeltaArray)
        console.print("mse: {}".format(mse))
        
        new_correlation.append({
            'correlationoject':correlationoject,
            'mse': mse,
            'HEMVectorMatrix': HEMVectorMatrix,
            'tableP' : tableP,
            'correlationDeltaMatrix': correlationDeltaMatrix,
            'mvgdCorrelationMatrix': mvgdCorrelationMatrix
        })

    #select the matrix with Minimum Mean Squared Error
    mse = 0
    selected_index = -1
    for idx,elm in enumerate(new_correlation):
        if elm['mse'] < mse or selected_index == -1:
            mse = elm['mse']
            selected_index = idx

    console.print('+++++++++++++++++++++')
    console.print('minimum mse: {}'.format(mse))
    console.print('selected_index: {}'.format(selected_index))
    console.print('HEMVectorMatrix: {}'.format(new_correlation[selected_index]['HEMVectorMatrix']))
    console.print(tableP)

    flatCorrelation = []
    for row in new_correlation[selected_index]['correlationDeltaMatrix']:
        for elm in row:
            flatCorrelation.append(elm)


    a = numpy.array(flatCorrelation)

    ## TODO: REGRESSIONI E CORRELAZIONI
    # PREDICTION USING CORRELAZIONI
    # RIMOZIONI COLONNE ALTA CORRELAZIONE, CHE POSSIAMO DERIVARE



    ## TODO: PLOTTING
    ## FIG. 1 percentile delta correlazione

    # FIG.2 AVG. MSE IN BASE A QUANTE ITERAZIONI DELLO STEP DI OTTIMIZZAZIONI ABBIAMO

    # FIG.3 CORRELAZIONE EMPIRICA QUADRATONI

    # FIG4 CORRELAZIONE MVGD QUADRATONI

    console.print('np.array(flatCorrelation): {}'.format(a))
    console.print('np.percentile(a, 20): {}'.format(numpy.percentile(a, 20)))
    console.print('np.percentile(a, 50): {}'.format(numpy.percentile(a, 50)))
    console.print('np.percentile(a, 70): {}'.format(numpy.percentile(a, 70)))
    console.print('np.percentile(a, 90): {}'.format(numpy.percentile(a, 90)))
    console.print('np.percentile(a, 99): {}'.format(numpy.percentile(a, 99)))

    

    # FIG.3 CORRELAZIONE EMPIRICA QUADRATONI
    df = pandas.DataFrame(new_correlation[selected_index]['mvgdCorrelationMatrix'], columns = [elm for elm in MUCH_BENCH_PMUS], index = [elm for elm in MUCH_BENCH_PMUS])
    console.print('df: {}'.format(df))

    plt.figure(figsize=(16,12))

    mask = numpy.triu(numpy.ones_like(df, dtype=bool))

    # Create a custom diverging palette
    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                                n=9, center="light", as_cmap=True)

    _ = sns.heatmap(df, mask=mask, center=0, annot=True,
             fmt='.2f', square=True, cmap=cmap)

    plt.show()


    dataframeCorrelationMatrix = pandas.DataFrame(correlationMatrix, columns = [elm for elm in MUCH_BENCH_PMUS], index = [elm for elm in MUCH_BENCH_PMUS])
    console.print('dataframeCorrelationMatrix: {}'.format(dataframeCorrelationMatrix))

    plt.figure(figsize=(16,12))

    mask = numpy.triu(numpy.ones_like(dataframeCorrelationMatrix, dtype=bool))

    # # Create a custom diverging palette
    # cmap = sns.diverging_palette(250, 15, s=75, l=40,
    #                             n=9, center="light", as_cmap=True)
    # Create a custom diverging palette
    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                                n=9, center="light", as_cmap=True)
    _ = sns.heatmap(dataframeCorrelationMatrix, mask=mask, center=0, annot=True,
             fmt='.2f', square=True, cmap=cmap)

    plt.show()

    # FIG4 CORRELAZIONE MVGD QUADRATONI

    #     bestCorrelation.append({
    #         'idx': i,
    #         'mvgdMSE': mvgdMSE,
    #         'HEMVectorMatrix' : HEMVectorMatrix,
    #         'mvgdCorrelationMatrix': mvgdCorrelationMatrix
    #     })
    # for elm in bestCorrelation:
    #     console.print(elm)


    ## VALUE PREDICTION FOR ESTIMATED 
    df = pandas.DataFrame(HEMVectorMatrix[:,index] for index in range(0,len(MUCH_BENCH_PMUS)))
    console.print(df)
    df.columns = [h for h in MUCH_BENCH_PMUS]
    console.print(df)

    y = df['l2d_cache_refill']
    X = df.drop(columns='l2d_cache_refill')


    # Linear Regression
    MLR = LinearRegression(
    copy_X = True,
    fit_intercept = True,
    n_jobs = 'auto',
    normalize = 'deprecated',
    positive = False
    )

    MLR.fit(X, y)

    y_pred = MLR.predict(X)
    MAE_MLR = mean_absolute_percentage_error(y, y_pred)

    #MLP Regression
    MLP = MLPRegressor(
    activation = 'relu',
    alpha = 0.0001,
    batch_size = 'auto',
    beta_1 = 0.9,
    beta_2 = 0.999,
    early_stopping = False,
    epsilon = 1e-08,
    hidden_layer_sizes = (100,),
    learning_rate = 'constant',
    learning_rate_init = 0.001,
    max_fun = 15000,
    max_iter = 200,
    momentum = 0.9,
    n_iter_no_change = 10,
    nesterovs_momentum = True,
    power_t = 0.5,
    random_state = None,
    shuffle = True,
    solver = 'adam',
    tol = 0.0001,
    validation_fraction = 0.1,
    verbose = False,
    warm_start = False
    )

    MLP.fit(
        X, 
        y
    )

    MLP.fit(X, y)

    y_pred = MLP.predict(X)

    MAE_MLP = mean_absolute_percentage_error(y, y_pred)

    #Random Forest Regression
    RF = RandomForestRegressor(
    bootstrap = True,
    ccp_alpha = 0.0,
    criterion = 'squared_error',
    max_depth = None,
    max_features = 'auto',
    max_leaf_nodes = None,
    max_samples = None,
    min_impurity_decrease = 0.0,
    min_samples_leaf = 1,
    min_samples_split = 2,
    min_weight_fraction_leaf = 0.0,
    n_estimators = 100,
    n_jobs = -1,
    oob_score = False,
    random_state = None,
    verbose = 0,
    warm_start = False
    )

    RF.fit(
        X, 
        y
    )

    RF.fit(X, y)

    y_pred = RF.predict(X)

    MAE_RF = mean_absolute_percentage_error(y, y_pred)

    models = ['MLR', 'MLP', 'RF']
    MAEs = [MAE_MLR, MAE_MLP, MAE_RF]
    df1 = pandas.DataFrame([models, MAEs]).T
    df1.columns = ['Model', 'MAE']
    console.print('df1: {}'.format(df1))


    ## TROVARE I 6 PMU CHE APPROSSIMANO MEGLIO IL RESTO DELLA DISTRUBUZIONE
    
    bestfit_df = pandas.DataFrame(new_correlation[selected_index]['mvgdCorrelationMatrix'], columns = MUCH_BENCH_PMUS, index = MUCH_BENCH_PMUS)
    console.print("bestfit_df: {}".format(bestfit_df))
    bestHEMS = {
        'hems' : [],
        'correlationSum': 0
    }

    for hem_list in itertools.combinations(MUCH_BENCH_PMUS, 6):
        window_columns = [elm for elm in MUCH_BENCH_PMUS if elm not in hem_list]
        window_matrix = bestfit_df.loc[hem_list,window_columns]
        # console.print("window_matrix: {}".format(window_matrix))
        # console.print("hem_list: {}".format(hem_list))

        localCorrelationSum = 0
        for col in window_columns:
            col_filter = window_matrix.loc[:,col]
            # console.print("col_filter: {}".format(col_filter))
            # console.print("col: {}".format(col))
            # console.print("window_matrix: {}".format(window_matrix))
            localCorrelationSum += col_filter.abs().max()
            # console.print("max_val: {}".format(max_val))
        
        if bestHEMS['correlationSum'] < localCorrelationSum:
            bestHEMS['correlationSum'] = localCorrelationSum
            bestHEMS['hems'] = hem_list
            console.print("++++++++++++++++++++++++++")
            console.print("localCorrelationSum: {}".format(localCorrelationSum))
            console.print("hem_list: {}".format(hem_list))

    console.print("best correlation FIT: {}".format(bestHEMS['correlationSum']))
    console.print("best HEMs: {}".format(bestHEMS['hems']))

    df= pandas.DataFrame(HEMVectorMatrix[:,index] for index in range(0,len(MUCH_BENCH_PMUS)))
    df.columns = [h for h in MUCH_BENCH_PMUS]
    console.print(df)

    #droppo altre colonne
    filteredOutColumns = [elm for elm in MUCH_BENCH_PMUS if elm not in bestHEMS['hems']]
    X= df.drop(columns=filteredOutColumns)

    predictions = []
    for hem in filteredOutColumns:
        y = df[hem]
        # Linear Regression
        MLR = LinearRegression(
        copy_X = True,
        fit_intercept = True,
        n_jobs = 'auto',
        normalize = 'deprecated',
        positive = False
        )

        MLR.fit(X, y)

        y_pred = MLR.predict(X)
        MAE_MLR = mean_absolute_percentage_error(y, y_pred)

        #MLP Regression
        MLP = MLPRegressor(
        activation = 'relu',
        alpha = 0.0001,
        batch_size = 'auto',
        beta_1 = 0.9,
        beta_2 = 0.999,
        early_stopping = False,
        epsilon = 1e-08,
        hidden_layer_sizes = (100,),
        learning_rate = 'constant',
        learning_rate_init = 0.001,
        max_fun = 15000,
        max_iter = 200,
        momentum = 0.9,
        n_iter_no_change = 10,
        nesterovs_momentum = True,
        power_t = 0.5,
        random_state = None,
        shuffle = True,
        solver = 'adam',
        tol = 0.0001,
        validation_fraction = 0.1,
        verbose = False,
        warm_start = False
        )

        MLP.fit(
            X, 
            y
        )

        # MLP.fit(X, y)

        y_pred = MLP.predict(X)

        MAE_MLP = mean_absolute_percentage_error(y, y_pred)

        #Random Forest Regression
        RF = RandomForestRegressor(
        bootstrap = True,
        ccp_alpha = 0.0,
        criterion = 'squared_error',
        max_depth = None,
        max_features = 'auto',
        max_leaf_nodes = None,
        max_samples = None,
        min_impurity_decrease = 0.0,
        min_samples_leaf = 1,
        min_samples_split = 2,
        min_weight_fraction_leaf = 0.0,
        n_estimators = 100,
        n_jobs = -1,
        oob_score = False,
        random_state = None,
        verbose = 0,
        warm_start = False
        )

        RF.fit(
            X, 
            y
        )

        RF.fit(X, y)

        y_pred = RF.predict(X)

        MAE_RF = mean_absolute_percentage_error(y, y_pred)

        predictions.append({
            'HEM': hem,
            'MLR': MAE_MLR,
            'MLP': MAE_MLP,
            'RF': MAE_RF
        })
    console.print('predictions: {}'.format(predictions))
    # models = ['HEM', 'MLR', 'MLP', 'RF']
    # MAEs = [hem ,MAE_MLR, MAE_MLP, MAE_RF]
    # df1 = pandas.DataFrame([models, MAEs]).T
    # df1.columns = ['Model', 'MAE']
    # console.print('df1: {}'.format(df1))


































    
    # #scatter plot for correlation
    # corr = plt.figure(num=3, figsize=[10, 10])
    # scatter_x = []
    # scatter_y = []
    # scatter_n = []
    # for index,pmu in enumerate(MUCH_BENCH_PMUS):
    #     for pair in PMU_STATISTICS[pmu]["p"]:
    #         scatter_x.append(index)
    #         scatter_y.append(pair["val"])
    #         scatter_n.append(pair["pair"])
    # plt.scatter(scatter_x,scatter_y)
    # plt.xticks(range(0,len(MUCH_BENCH_PMUS)), MUCH_BENCH_PMUS,
    #    rotation=20)  # Set text labels and properties.
    # for i, txt in enumerate(scatter_n):
    #     plt.annotate(txt, (scatter_x[i],scatter_y[i]))
    # plt.rcParams["figure.figsize"] = (20,20)
    # corr.show()


    # fig = plt.figure(num=2)
    # ax = plt.axes(projection='3d')
    # scatter_x = []
    # scatter_y = []
    # scatter_n = []
    # for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
    #     for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
    #         if index1 == index2:
    #             continue #same PMU
    #         for pair in PMU_STATISTICS[pmu1]["p"]:
    #             if(pair["pair"] == pmu2):
    #                 scatter_x.append(index1)
    #                 scatter_y.append(index2)
    #                 scatter_n.append(pair["val"])

    # ax.scatter3D(scatter_x, scatter_y, scatter_n);    
    # ax.legend()
    # plt.xticks(range(0,len(MUCH_BENCH_PMUS)), MUCH_BENCH_PMUS,
    #    rotation=20)
    # plt.yticks(range(0,len(MUCH_BENCH_PMUS)), MUCH_BENCH_PMUS)
    # fig.show()
    # plt.show()





    #pts = np.random.multivariate_normal(0^,  Σˆ0, size=numeri rilevamenti hi HEM)

    
def initalizeExperimentObject(experiment):
    global EXPERIMENTS_RESULTS_TABLE
    EXPERIMENTS_RESULTS_TABLE.append({
        "pmus": experiment,
        "data": []
    })


def collectMUCHValues(report, indexExperiment, experiment):
    global table, RUN_COUNTER, MUCH_BENCH_PMUS

    # write to aux file for line readings buffer
    fileObject = open("aux.temp", 'w')
    fileObject.write(report)
    fileObject.close()
    
    fileObject = open("aux.temp", 'r')
    fileLines = fileObject.readlines()

    pmuReportList = []

    for line in fileLines:
        for idx in range(0, len(experiment)):
            pmuReportList.append({
                    "pmu": experiment[idx],
                    "events": line.split('/')[idx]
                })
    EXPERIMENTS_RESULTS_TABLE[indexExperiment]["data"].append(pmuReportList)
    fileObject.close()

def loadObjects():
    global PMU_STATISTICS, MUCH_BENCH_PMUS, console, PMU_GROUPED_HI, EXPERIMENTS_LIST,EXPERIMENTS_RESULTS_TABLE

    with open(args.load, 'rb') as f:
        data = pickle.load(f)
    
    EXPERIMENTS_LIST = data['EXPERIMENTS_LIST']
    MUCH_BENCH_PMUS = data['PMU_SUPPORTED']
    EXPERIMENTS_RESULTS_TABLE = data['EXPERIMENTS_RESULTS_TABLE']
    console = Console()
    logging.debug(EXPERIMENTS_LIST)
    logging.debug(MUCH_BENCH_PMUS)
    logging.debug(EXPERIMENTS_RESULTS_TABLE)


def checkComplementaryPMUIterations(chosenMuchPmus, secondaryPMU):
    global MUCH_EXECUTED_ITERATION
    for pmu in chosenMuchPmus:
        try:
            if secondaryPMU in MUCH_EXECUTED_ITERATION[pmu]:
                return False
        except KeyError:
            MUCH_EXECUTED_ITERATION[pmu] = []
    return True


if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                       '--sudo',
                       dest="sudo",
                       action='store_true',
                       default=False,
                       help='exec command with sudo capabilities')

    parser.add_argument('-d',
                       '--debug',
                       dest="debug",
                       action='store_true',
                       default=False,
                       help='debug prints')

    parser.add_argument('-w',
                       '--write',
                       dest="write",
                       type=str,
                       help='write benchmarks to disk')

    parser.add_argument('-l',
                       '--load',
                       dest="load",
                       type=str,
                       help='load benchmarks results from disk')

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if args.debug == True else logging.ERROR)
    print('CHECK!!!!!!!')
    print('https://developer.arm.com/tools-and-software/simulation-models/cycle-models/knowledge-articles/using-the-arm-performance-monitor-unit-linux-driver')
    # sudo perf stat -I 1000 -e cycles -a sleep 5
    # https://man7.org/linux/man-pages/man1/perf-stat.1.html
    # implement cycle based calculations ??????????? 
    if(not args.load):
        main()
    else:
        loadObjects()
        drawingData()
