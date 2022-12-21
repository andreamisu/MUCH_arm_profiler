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

def bestModelSelection(dataframe_pred, df, bestHEMS, MUCH_BENCH_PMUS):
    #switch cases are only available on python 3.10 *sigh*
    X= df.drop(columns=[elm for elm in MUCH_BENCH_PMUS if elm not in bestHEMS])
    y= df.drop(columns=[elm for elm in MUCH_BENCH_PMUS if elm in bestHEMS])

    minErrorML = 9999999
    selectedModel = ''
    for col in dataframe_pred.columns:
        col_filter = dataframe_pred.loc[:,col]
        if col_filter.max() < minErrorML:
            minErrorML = col_filter.max()
            selectedModel = col

    print(selectedModel)
    print(minErrorML)
    if selectedModel == 'MLP':
        model = MLPRegressor(
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
    elif selectedModel == 'MLR':
        model = LinearRegression(
            copy_X = True,
            fit_intercept = True,
            n_jobs = 'auto',
            normalize = 'deprecated',
            positive = False
            )
       
    elif selectedModel == 'RF':
        model = RandomForestRegressor(
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
    
    else:
        print('unknown best model, shutting down.')
        exit()
    model.fit(X, y)
    return {
        'model': selectedModel,
        'obj': model
    }

        

    
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
                    logging.debug((line))
                    pmu = line.replace("\n","").split("/")[1]
                    perfList.append(pmu)
                    RAW_PMU[pmu] = line.replace("\n","").split("/")[2]

    logging.debug(("numbers of fetchable PMU events: %d" % (len(perfList))))
    logging.debug((",      ".join(perfList)))
    logging.debug(("numbers of PMU events each run: " + str(PMU_STEPS)))
    
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
    if(len(RUNS_FAILED) > 0):
        console.print("Failed runs: " + str(RUNS_FAILED), style="red")
        exit()
    # Algorithm for best allocation on given PMUs
    console.print("number of HEMs available for MUCH evaluation: %d" % (len(MUCH_BENCH_PMUS)))
    console.print(MUCH_BENCH_PMUS)
    MUCH_BENCH_PMUS = pmu_allocation(MUCH_BENCH_PMUS, ALLOCABLE_PMUS)


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
    console.print("sub-experiments: \n")
    for e in EXPERIMENTS_LIST:
        console.print(e)

    # starts experiments
    for index in range(0,len(EXPERIMENTS_LIST)):
        initalizeExperimentObject(EXPERIMENTS_LIST[index])
        pmuSelected = ''
        for idx, val in enumerate(EXPERIMENTS_LIST[index]):
            pmuSelected += RAW_PMU[val] if idx == 0 else "/"+RAW_PMU[val]
        cmdBench = ["sudo", PERF_COMMAND if args.sudo else PERF_COMMAND , pmuSelected]
        for i in range(0, MUCH_RUNS):
            with console.status("Benchmark for experiment # {} \n {} \n {} / {} runs".format(index+1, ",".join(EXPERIMENTS_LIST[index]), i, MUCH_RUNS)):
                results = subprocess.run(cmdBench, text=True, capture_output=True)
                if(results.returncode == 0):
                    collectMUCHValues(results.stdout, index, EXPERIMENTS_LIST[index])
                else:
                    console.print("Benchmark exited with {} status code, shutting down.".format(results.returncode), style='blink')
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
        logging.debug(('{}={}'.format(h,PMU_GROUPED_HI[h])))
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
        for idx,elm in enumerate(PMU_GROUPED_HI[h]):
            PMU_MAPPED[h][elm] = pmu1_argsort[idx]

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
                    x = -1
                    y = -1
                    for subexp in data:
                        if subexp["pmu"] == pmu_couple[0]:
                            x = int(subexp["events"].replace(",",""))
                        if subexp["pmu"] == pmu_couple[1]:
                            y = int(subexp["events"].replace(",",""))
                    if x == -1 or y == -1:
                        console.print("Error: didn't find empirical correlation, shutting down")
                        exit()
                    pmu1.append(x)
                    pmu2.append(y)
                # pearson correlation
                p, pvalue = pearsonr(pmu1, pmu2)
                logging.debug('{} + {} correlation: {}'.format(pmu_couple[0], pmu_couple[1],p))
                logging.debug('pmu1: {}'.format(pmu1))
                logging.debug('pmu2: {}'.format(pmu2))

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
                logging.debug('%s + %s correlation: %f' % (pmu_couple[0], pmu_couple[1],p_mapped))


                ## tenere queste variance e std per creazione correlation matrix MVGD
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
        logging.debug("pmu1: %s" % pmu1)
        correlationLine = []
        for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
            logging.debug("pmu2: %s" % pmu2)
            if pmu1==pmu2:
                logging.debug("same")
                correlationLine.append(float(1)) #correlation between same values is 1
            else:
                for corr in PMU_STATISTICS[pmu2]['p']:
                    if corr['pair'] == pmu1:
                        correlationLine.append(corr['val'])
        correlationMap.append(correlationLine)
    console.print("++++++++++++++++++++++++++++++++++")
    logging.debug(correlationMap)
    correlationMatrix = numpy.array(correlationMap, dtype=numpy.float64)
    console.print("correlation matrix: {}".format(correlationMatrix))
    console.print("++++++++++++++++++++++++++++++++++")

    #Covariance matrix Ʃˆ
    mean_array = []
    for pmu in MUCH_BENCH_PMUS:
        mean_array.append(PMU_STATISTICS[pmu]['u'])

    covarianceMap = []
    for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
        logging.debug("pmu1: %s" % pmu1)
        covarianceLine = []
        for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
            logging.debug("pmu2: %s" % pmu2)
            if pmu1==pmu2:
                covarianceLine.append(PMU_STATISTICS[pmu2]['o2'])
            else:
                for corr in PMU_STATISTICS[pmu1]['o_pair']:
                    if corr['pair'] == pmu2:
                        covarianceLine.append(corr['val'])
                        continue
        covarianceMap.append(covarianceLine)
    console.print("++++++++++++++++++++++++++++++++++")
    console.print("covariance matrix:")
    console.print("++++++++++++++++++++++++++++++++++")
    covarianceMatrix = numpy.array(covarianceMap, dtype = numpy.float64)
    console.print(covarianceMatrix)

    #Application of copula theory
    for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
        num = len(PMU_GROUPED_HI[pmu1])
        steps,dist = numpy.linspace(0,1,num+1,endpoint=False, dtype=numpy.float64, retstep=True)
        steps = steps.tolist()
        steps.remove(0)
        meanSteps = numpy.mean(steps)
        logging.debug("num is %d" % num)
        logging.debug("steps are %s" % str(steps))

        # A uniform sample can be transformed into a
        # Gaussian sample by applying the inverse function 
        # of the cumulative distribution function of a standard Gaussian distribution, Φ

        # Percent point function (inverse of cdf — percentiles).
        mvgdSampled = list(map(lambda elm: norm.ppf(elm, loc=0, scale=1), steps)) 
        logging.debug(mvgdSampled)
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
        logging.debug("ppf0 -> " + str(PMU_STATISTICS[pmu_couple[0]]['ppf']))
        logging.debug("ppf1 -> " + str(PMU_STATISTICS[pmu_couple[1]]['ppf']))
        logging.debug("covariance : {}".format(numpy.cov([PMU_STATISTICS[pmu_couple[0]]['ppf']['val'],PMU_STATISTICS[pmu_couple[0]]['ppf']['val']])))
        logging.debug("p : {}".format(p))
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
        logging.debug("pmu1: %s" % pmu1)
        correlationLine = []
        for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
            logging.debug("pmu2: %s" % pmu2)
            if pmu1==pmu2:
                correlationLine.append(float(1)) #correlation between same values is 1
            else:
                for corr in PMU_STATISTICS[pmu2]['p']:
                    if corr['pair'] == pmu1:
                        correlationLine.append(corr['val_mapped'])
        correlationMap.append(correlationLine)
    console.print("++++++++++++++++++++++++++++++++++")
    gaussianCorrelationMatrix = numpy.array(correlationMap, dtype=numpy.float64)
    console.print("gaussianCorrelationMatrix: {}".format(gaussianCorrelationMatrix))
    console.print("++++++++++++++++++++++++++++++++++")

    #Covariance matrix Ʃˆ0
    gaussianCovarianceMap = []
    for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
        logging.debug("pmu1: %s" % pmu1)
        covarianceLine = []
        for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
            logging.debug("pmu2: %s" % pmu2)
            if pmu1==pmu2:
                covarianceLine.append(PMU_STATISTICS[pmu1]['o2_mapped'])
            else:
                for corr in PMU_STATISTICS[pmu1]['o_pair']:
                    if corr['pair'] == pmu2:
                        covarianceLine.append(corr['val_mapped'])
                        continue
        gaussianCovarianceMap.append(covarianceLine)
    console.print("++++++++++++++++++++++++++++++++++")
    gaussianCovarianceMatrix = numpy.matrix(gaussianCovarianceMap, dtype = numpy.float64)
    console.print("gaussianCovarianceMatrix:\n{}".format(gaussianCovarianceMatrix))
    console.print("++++++++++++++++++++++++++++++++++")

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
                logging.debug("len MUCH_BENCH: {}".format(MUCH_BENCH_PMUS))
                logging.debug("index1: {}".format(index1))
                logging.debug("index2: {}".format(index2))
                if index2 <= index1:
                    continue #same PMU
                pmu1Values = HEMVectorMatrix[index1,:]
                pmu2Values = HEMVectorMatrix[index2,:]
                mvgdP, _ = pearsonr(pmu1Values, pmu2Values)
                logging.debug(mvgdP)
                for pair in PMU_STATISTICS[pmu1]["p"]:
                    if(pair["pair"] == pmu2):
                        empiricalP = pair["val"]
                        deltaP = abs(mvgdP - empiricalP)
                        logging.debug('mvgdP: {}'.format(mvgdP))
                        logging.debug('empiricalP: {}'.format(empiricalP))
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
            logging.debug("pmu1: %s" % pmu1)
            correlationLine = []
            for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
                logging.debug("pmu2: %s" % pmu2)
                if pmu1==pmu2:
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
        console.print("++++++++++++++++++++++++++++++++++")

        mse = 0
        logging.debug("correlationMatrix: {}".format(correlationMatrix))
        logging.debug("mvgdCorrelationMatrix: {}".format(mvgdCorrelationMatrix))
        correlationDeltaMatrix = []
        for index_i in range(0,len(MUCH_BENCH_PMUS)):
            correlationDeltaArray = []
            for index_j in range(0,len(MUCH_BENCH_PMUS)):
                logging.debug("correlationMatrix[{}][{}]: {}".format(index_i, index_j, correlationMatrix[index_i][index_j]))
                logging.debug("correlationMatrix[{}][{}]: {}".format(index_i, index_j, mvgdCorrelationMatrix[index_i][index_j]))
                correlationDeltaArray.append(abs(correlationMatrix[index_i][index_j] - mvgdCorrelationMatrix[index_i][index_j]))
                mse += math.pow(correlationMatrix[index_i][index_j] - mvgdCorrelationMatrix[index_i][index_j], 2)
            correlationDeltaMatrix.append(correlationDeltaArray)
        logging.debug("mse: {}".format(mse))
        
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
    console.print(tableP)

    flatCorrelation = []
    for row in new_correlation[selected_index]['correlationDeltaMatrix']:
        for elm in row:
            flatCorrelation.append(elm)


    a = numpy.array(flatCorrelation)
    logging.debug('np.array(flatCorrelation): {}'.format(a))
    logging.debug('np.percentile(a, 20): {}'.format(numpy.percentile(a, 20)))
    logging.debug('np.percentile(a, 50): {}'.format(numpy.percentile(a, 50)))
    logging.debug('np.percentile(a, 70): {}'.format(numpy.percentile(a, 70)))
    logging.debug('np.percentile(a, 90): {}'.format(numpy.percentile(a, 90)))
    logging.debug('np.percentile(a, 99): {}'.format(numpy.percentile(a, 99)))

    

    #plotting mvgd correlation matrix
    df = pandas.DataFrame(new_correlation[selected_index]['mvgdCorrelationMatrix'], columns = [elm for elm in MUCH_BENCH_PMUS], index = [elm for elm in MUCH_BENCH_PMUS])
    console.print('df:\n{}'.format(df))
    plt.figure(figsize=(16,12))
    mask = numpy.triu(numpy.ones_like(df, dtype=bool))
    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                                n=9, center="light", as_cmap=True)
    _ = sns.heatmap(df, mask=mask, center=0, annot=True,
             fmt='.2f', square=True, cmap=cmap)
    plt.tight_layout(pad=0.1)
    plt.savefig("mvgd_correlation_matrix.svg", format="svg")
    plt.show()

    #plotting empirical correlation matrix
    dataframeCorrelationMatrix = pandas.DataFrame(correlationMatrix, columns = [elm for elm in MUCH_BENCH_PMUS], index = [elm for elm in MUCH_BENCH_PMUS])
    plt.figure(figsize=(16,12))
    mask = numpy.triu(numpy.ones_like(dataframeCorrelationMatrix, dtype=bool))
    cmap = sns.diverging_palette(250, 15, s=75, l=40,
                                n=9, center="light", as_cmap=True)
    _ = sns.heatmap(dataframeCorrelationMatrix, mask=mask, center=0, annot=True,
             fmt='.2f', square=True, cmap=cmap)
    plt.tight_layout(pad=0.1)
    plt.savefig("empirical_correlation_matrix.svg", format="svg")
    plt.show()


    with open(PERF_LIST_FILENAME) as f:
        for line in f.readlines():
            if "#" not in line: #filter out comments
                logging.debug(line)
                pmu = line.replace("\n","").split("/")[1]
                RAW_PMU[pmu] = line.replace("\n","").split("/")[2]


    ## HEMs cluster creation using 6 slots


    bestHEMS = {
        'hems' : [],
        'correlationSum': 0
    }
    bestfit_df = pandas.DataFrame(new_correlation[selected_index]['mvgdCorrelationMatrix'], columns = MUCH_BENCH_PMUS, index = MUCH_BENCH_PMUS)
    for hem_list in itertools.combinations(MUCH_BENCH_PMUS, 6):
        window_columns = [elm for elm in MUCH_BENCH_PMUS if elm not in hem_list]
        window_matrix = bestfit_df.loc[hem_list,window_columns]
        logging.debug("window_matrix: {}".format(window_matrix))
        logging.debug("hem_list: {}".format(hem_list))

        localCorrelationSum = 0
        for col in window_columns:
            col_filter = window_matrix.loc[:,col]
            localCorrelationSum += col_filter.abs().max()
        
        if bestHEMS['correlationSum'] < localCorrelationSum:
            bestHEMS['correlationSum'] = localCorrelationSum
            bestHEMS['hems']  = hem_list
            console.print("++++++++++++++++++++++++++")
            console.print("localCorrelationSum: {}".format(localCorrelationSum))
            console.print("hem_list: {}".format(hem_list))
            console.print("++++++++++++++++++++++++++")

    console.print("best correlation FIT: {}".format(bestHEMS['correlationSum']))
    console.print("best HEMs: {}".format(bestHEMS['hems']))
    



    #VALIDATION: FIND 5 BEST FITTING HEMs TO USE THEM IN EXPERIMENTS AND UNDERSTAND MEAN ERROR ON PREDICTIONS
    bestLocalHEMS = {
        'hems' : [],
        'correlationSum': 0
    }
    with console.status("Finding HEMs cluster using 5 HEMs"):
        for hem_list in itertools.combinations(MUCH_BENCH_PMUS, 5):
            window_columns = [elm for elm in MUCH_BENCH_PMUS if elm not in hem_list]
            window_matrix = bestfit_df.loc[hem_list,window_columns]
            localCorrelationSum = 0
            for col in window_columns:
                col_filter = window_matrix.loc[:,col]
                localCorrelationSum += col_filter.abs().max()
            
            if bestLocalHEMS['correlationSum'] < localCorrelationSum:
                bestLocalHEMS['correlationSum'] = localCorrelationSum
                bestLocalHEMS['hems'] = hem_list
                logging.debug("++++++++++++++++++++++++++")
                logging.debug("localCorrelationSum: {}".format(localCorrelationSum))
                logging.debug("hem_list: {}".format(hem_list))

    console.print("best correlation FIT: {}".format(bestLocalHEMS['correlationSum']))
    console.print("best HEMs: {}".format(bestLocalHEMS['hems']))

    df= pandas.DataFrame(HEMVectorMatrix[:,index] for index in range(0,len(MUCH_BENCH_PMUS)))
    df.columns = [h for h in MUCH_BENCH_PMUS]
    filteredOutColumns = [elm for elm in MUCH_BENCH_PMUS if elm not in bestLocalHEMS['hems']]
    X= df.drop(columns=filteredOutColumns)
    predictions = []
    for hem in filteredOutColumns:
        with console.status("testing out ML models using {}".format(hem)):
            y = df[hem]
            logging.debug('X: {}'.format(X))
            logging.debug('y: {}'.format(y))

            ##EXPERIMENT 
            pmuSelected = ''
            pmuReportList = []
            for idx, val in enumerate(bestLocalHEMS['hems']):
                pmuSelected += RAW_PMU[val] if idx == 0 else "/"+RAW_PMU[val]
            pmuSelected += "/"+RAW_PMU[hem]
            logging.debug('pmuSelected: {}'.format(pmuSelected))
            logging.debug('pmuSelected.split(): {}'.format(pmuSelected.split('/')))
            cmdBench = ["sudo", PERF_COMMAND if args.sudo else PERF_COMMAND , pmuSelected]
            for i in range(0, 10):
                results = subprocess.run(cmdBench, text=True, capture_output=True)
                if(results.returncode == 0):
                        logging.debug('results.stdout: {}'.format(results.stdout))                    
                        pmuReportList.append(results.stdout.split('/'))
                else:
                    logging.debug("unexpected error")
                    return -1
            exp_df = pandas.DataFrame(pmuReportList)
            localHems = list(bestLocalHEMS['hems'])
            localHems.append(hem)
            logging.debug('localHems: {}'.format(localHems))
            exp_df.columns = [localHems]
            logging.debug('exp_df: {}'.format(exp_df))

            X_exp= exp_df.drop(columns=hem)
            Y_exp= exp_df.drop(columns=list(bestLocalHEMS['hems']))

            logging.debug('exp_df:{}'.format(exp_df))
            logging.debug('X_exp:{}'.format(X_exp))
            logging.debug('Y_exp:{}'.format(Y_exp))


            # Linear Regression
            MLR = LinearRegression(
            copy_X = True,
            fit_intercept = True,
            n_jobs = 'auto',
            normalize = 'deprecated',
            positive = False
            )

            MLR.fit(X, y)

            y_pred = MLR.predict(X_exp)
            MAE_MLR = mean_absolute_percentage_error(Y_exp, y_pred)
            
            logging.debug('MLR')
            logging.debug('Y_exp:{}'.format(Y_exp))
            logging.debug('y_pred:{}'.format(y_pred))
            logging.debug('MAE_MLR:{}'.format(MAE_MLR))


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

            y_pred = MLP.predict(X_exp)

            MAE_MLP = mean_absolute_percentage_error(Y_exp, y_pred)
            logging.debug('MLP')
            logging.debug('Y_exp:{}'.format(Y_exp))
            logging.debug('y_pred:{}'.format(y_pred))
            logging.debug('MAE_MLP:{}'.format(MAE_MLP))

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

            y_pred = RF.predict(X_exp)

            MAE_RF = mean_absolute_percentage_error(Y_exp, y_pred)
            logging.debug('RF')
            logging.debug('Y_exp:{}'.format(Y_exp))
            logging.debug('y_pred:{}'.format(y_pred))
            logging.debug('MAE_RF:{}'.format(MAE_RF))

            predictions.append({
                'MLR': MAE_MLR,
                'MLP': MAE_MLP,
                'RF': MAE_RF
            })

    dataframe_pred= pandas.DataFrame(predictions, columns=['MLR', 'MLP', 'RF'], index=filteredOutColumns)
    console.print('dataframe_pred:\n{}'.format(dataframe_pred))

    #we save the ML model that has minimum absolute error
    model = bestModelSelection(dataframe_pred, df, bestHEMS['hems'], MUCH_BENCH_PMUS)

    data = {
        'ML_MODEL': model,
        'BESTFIT_HEMS' : bestHEMS['hems'],
        'MUCH_BENCH_PMUS': MUCH_BENCH_PMUS,
        'HEMVectorMatrix': HEMVectorMatrix
    }

    with open('./{}.much'.format((args.model if args.model else 'benchmark')), 'wb') as f:
        pickle.dump(data, f)
    console.print("Experiments done! data is written in './{}.much file.".format((args.model if args.model else 'benchmark')))

    
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

    parser.add_argument('-m',
                       '--model',
                       dest="model",
                       type=str,
                       help='Model name for .much export file')


    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if args.debug == True else logging.ERROR)
    if(not args.load):
        main()
    else:
        loadObjects()
        drawingData()
