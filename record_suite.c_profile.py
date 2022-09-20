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
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from mpl_toolkits import mplot3d
import pickle 


BENCHMARK_STATISTICS_FILE = './benchmark_statistics.dump'
BENCHMARK_PMU_FILE = './benchmark_pmus.dump'
PERF_COMMAND = "./profiling"
PERF_LIST_FILENAME = "./pmu_lists/perf.armA53.list"
PERF_REPORT_FILENAME = "benchmarks.out"
RUN_COUNTER = 0
PMU_STEPS = 6
RAW_PMU = {} #key: pmu name / value: hex raw pmu
RUNS_FAILED = []
EVENTS_THRESHOLD = 20000
MUCH_BENCH_PMUS = []
EXPERIMENTS_LIST = []
ALLOCABLE_PMUS = 6
MUCH_RUNS = 50 # 30 is the minimum suggestabele from the paper in order to use Central Limit Theory. Higher the number, More precise the measurement of correlations.

PMU_GROUPED_HI = {}
PMU_STATISTICS = {} #key: pmu name


EXPERIMENTS_RESULTS_TABLE = []  #main obejct in which store results from experiments
# {
#     index: index of experiments,
#     pmus: [pmu names] of experiments,
#     data: [] array with every experiment data inside
# }

EFFICIENT_PMUS_ALLOCATION = 6 # EPA => # MUCH_BENCH_PMUS - (EPA + (EPA-1 * EPA-1)) =(or near) 0      so in the case of 30 allocable valid MUCH PMUs, EPA is 6 as 6 + 5 * 5 = 31     this is mostly helpful for correlation runs

MUCH_EXECUTED_ITERATION = {} #each index i refers to MUCH_BENCH_PMUS[i], and basically contains which PMUs has been already checked with the given event monitor.

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
    # console.print("Raw PMU: " + str(RAW_PMU))
    
    for x in range(PMU_STEPS, len(perfList), PMU_STEPS):
        pmuSelected = ''
        for idx, val in enumerate(perfList[x-PMU_STEPS:x]):
            pmuSelected += RAW_PMU[val] if idx == 0 else "/"+RAW_PMU[val]
        # console.print(pmuSelected)
        cmdBench = ["sudo", PERF_COMMAND if args.sudo else PERF_COMMAND , pmuSelected]
        with console.status("doing benchmarks on " + pmuSelected.replace(",",", ") + "..."):
            # console.print(cmdBench)
            results = subprocess.run(cmdBench, text=True, capture_output=True)
            if(results.returncode == 0):
                console.print(results.stdout)
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
    for pivotPMU in MUCH_BENCH_PMUS:
        try:
            len(MUCH_EXECUTED_ITERATION[pivotPMU])
        except KeyError:
            MUCH_EXECUTED_ITERATION[pivotPMU] = []
        while len(set(MUCH_EXECUTED_ITERATION[pivotPMU])) < len(MUCH_BENCH_PMUS):
            console.print("pivot: %s" % pivotPMU)
            console.print("MUCH_EXECUTED_ITERATION[pivotPMU]: %s" % MUCH_EXECUTED_ITERATION[pivotPMU])
            console.print("len: %d su %d" % (len(MUCH_EXECUTED_ITERATION[pivotPMU]), len(MUCH_BENCH_PMUS)))
            
            chosenMuchPmus = [pivotPMU]

            #TODO: Shall we take into account also multiple iteration on tuple of PMUs repeteing?
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
        console.print("should be FULL >> len: %d su %d" % (len(MUCH_EXECUTED_ITERATION[pivotPMU]), len(MUCH_BENCH_PMUS)))
        console.print("MUCH_EXECUTED_ITERATION[%s]: %s" % (pivotPMU, MUCH_EXECUTED_ITERATION[pivotPMU]))
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
            with console.status("Benchmark for experiment # %d \n %s \n %d / %d runs" % (index+1, ",".join(EXPERIMENTS_LIST[index]), i, MUCH_RUNS)):
                results = subprocess.run(cmdBench, text=True, capture_output=True)
                if(results.returncode == 0):
                    collectMUCHValues(results.stdout, index, EXPERIMENTS_LIST[index])
                else:
                    console.print("unexpected error")
                    return -1

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

            PMU_STATISTICS[h] = {
                "u": numpy.mean(PMU_GROUPED_HI[h]),
                "o": numpy.std(PMU_GROUPED_HI[h]), #standard deviation of hi subexperiment in PMU_GROUPED_HI[h] on u mean
                "o2": numpy.var(PMU_GROUPED_HI[h]) #variance of hi subexperiment in PMU_GROUPED_HI[h] on u mean
            }

            console.print("PMU_GROUPED_HI : %s" % (PMU_GROUPED_HI[h]))
            console.print("PMU expectedValue (u) value: %s" % str(PMU_STATISTICS[h]["u"]))
            console.print("PMU variance (o) value: %s" % str(PMU_STATISTICS[h]["o"]))
            console.print("PMU variance (o2) value: %s" % str(PMU_STATISTICS[h]["o2"]))

    # empirical correlation between 2 hems  ρˆij 
    for pmu_couple in itertools.combinations(MUCH_BENCH_PMUS, 2):
        for index in range(0,len(EXPERIMENTS_LIST)):
            #experiment in which both are present
            if pmu_couple[0] in EXPERIMENTS_LIST[index] and pmu_couple[1] in EXPERIMENTS_LIST[index]:
                pmu1 = []
                pmu2 = []
                p = 0

                try:
                    len(PMU_STATISTICS[pmu_couple[0]]["o_pair"])
                except KeyError:
                    PMU_STATISTICS[pmu_couple[0]]["o_pair"] = []
                    PMU_STATISTICS[pmu_couple[0]]["p"] = []

                try:
                    len(PMU_STATISTICS[pmu_couple[1]]["o_pair"])
                except KeyError:
                    PMU_STATISTICS[pmu_couple[1]]["o_pair"] = []
                    PMU_STATISTICS[pmu_couple[1]]["p"] = []

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
                p, _ = pearsonr(pmu1, pmu2)
                console.print('%s + %s correlation: %f' % (pmu_couple[0], pmu_couple[1],p))

                PMU_STATISTICS[pmu_couple[0]]["o_pair"].append({
                    "pair" : pmu_couple[1],
                    "val": p * PMU_STATISTICS[pmu_couple[0]]["o"] * PMU_STATISTICS[pmu_couple[1]]["o"]
                })


                PMU_STATISTICS[pmu_couple[1]]["o_pair"].append({
                    "pair" : pmu_couple[0],
                    "val": p * PMU_STATISTICS[pmu_couple[0]]["o"] * PMU_STATISTICS[pmu_couple[1]]["o"]
                })

                PMU_STATISTICS[pmu_couple[0]]["p"].append({
                    "pair" : pmu_couple[1],
                    "val": p
                })

                PMU_STATISTICS[pmu_couple[1]]["p"].append({
                    "pair" : pmu_couple[0],
                    "val": p
                })

                console.print("%s + %s:\no_pair > %s \np: %s" % (pmu_couple[0], pmu_couple[1], str(p * PMU_STATISTICS[pmu_couple[0]]["o"] * PMU_STATISTICS[pmu_couple[1]]["o"]), str(p)))
                #TODO: calculus empirical correlation matrix  S^

                #http://users.stat.umn.edu/~helwig/notes/datamat-Notes.pdf

    if(args.write):
        file_pi = open(BENCHMARK_STATISTICS_FILE, 'wb') 
        pickle.dump(PMU_STATISTICS, file_pi)
        file_pi2 = open(BENCHMARK_PMU_FILE, 'wb') 
        pickle.dump(MUCH_BENCH_PMUS, file_pi2)

    drawingData()

    

def drawingData():
    global PMU_STATISTICS, MUCH_BENCH_PMUS, console

    console.print(PMU_STATISTICS)
    console.print(MUCH_BENCH_PMUS)
    #Correlation matrix S
    correlationMap = []
    for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
        console.print("pmu1: %s" % pmu1)
        correlationLine = []
        for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
            flag=False
            console.print("pmu2: %s" % pmu2)
            if pmu1==pmu2:
                console.print("same")
                flag=True
                correlationLine.append(float(1)) #correlation between same values is 1
            else:
                for corr in PMU_STATISTICS[pmu2]['p']:
                    if corr['pair'] == pmu1:
                        correlationLine.append(corr['val'])
                        flag=True
                if(not flag):
                    console.print("NOT FOUND: %s, %s" %(pmu1, pmu2))
        correlationMap.append(correlationLine)
    console.print(correlationMap)
    correlationMatrix = numpy.matrix(correlationMap, dtype=float)
    console.print(correlationMatrix)
    plt.matshow(correlationMatrix)

    #Covariance matrix Ʃ
    covarianceMap = []
    for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
        console.print("pmu1: %s" % pmu1)
        covarianceLine = []
        for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
            console.print("pmu2: %s" % pmu2)
            if pmu1==pmu2:
                covarianceLine.append(math.pow(PMU_STATISTICS[pmu2]['o'], 2))
            else:
                for corr in PMU_STATISTICS[pmu2]['o_pair']:
                    if corr['pair'] == pmu1:
                        covarianceLine.append(corr['val'])
                        continue
        covarianceMap.append(covarianceLine)
    covarianceMatrix = numpy.matrix(covarianceMap, dtype=object)
    console.print(covarianceMatrix)
    



    #scatter plot for correlation
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
    # plt.show()


    fig = plt.figure()
    ax = plt.axes(projection='3d')
    scatter_x = []
    scatter_y = []
    scatter_n = []
    for index1,pmu1 in enumerate(MUCH_BENCH_PMUS):
        for index2,pmu2 in enumerate(MUCH_BENCH_PMUS):
            if index1 == index2:
                continue #same PMU
            for pair in PMU_STATISTICS[pmu1]["p"]:
                if(pair["pair"] == pmu2):
                    scatter_x.append(index1)
                    scatter_y.append(index2)
                    scatter_n.append(pair["val"])

    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens');    
    ax.legend()
    ax.xticks(range(0,len(MUCH_BENCH_PMUS)), MUCH_BENCH_PMUS,
       rotation=20)
    ax.yticks(range(0,len(MUCH_BENCH_PMUS)), MUCH_BENCH_PMUS,
       rotation=20)
    ax.show()
    
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
    global PMU_STATISTICS, MUCH_BENCH_PMUS, console

    file_pi1 = open(BENCHMARK_STATISTICS_FILE, 'rb') 
    PMU_STATISTICS = pickle.load(file_pi1)
    file_pi2 = open(BENCHMARK_PMU_FILE, 'rb') 
    MUCH_BENCH_PMUS = pickle.load(file_pi2)
    console = Console()


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
                       action='store_true',
                       default=False,
                       help='write benchmarks to disk')

    parser.add_argument('-l',
                       '--load',
                       dest="load",
                       action='store_true',
                       default=False,
                       help='load benchmarks from disk')
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if args.debug == True else logging.ERROR)
    if(not args.load):
        main()
    else:
        loadObjects()
        drawingData()
