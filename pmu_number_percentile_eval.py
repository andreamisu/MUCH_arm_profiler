from rich.table import Table
from rich.console import Console
from datetime import datetime
import subprocess
import argparse
from utils import *



PERF_LIST_FILENAME = "perf.rasp4.list"
PERF_REPORT_FILENAME = "percentile.out"
RUN_COUNTER = 0
COMMON_PMUS = ["cpu_cycles"]
PMU_STEPS = [5, 10, 15, 20, 30, 40]
RUNS_FAILED = []


def writeLogsFile(out, pmuSelected, pmuNumbers):
    global table, RUN_COUNTER
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
            print(str(pmuData))
            if "supported" not in str(pmuData[1]):
                table.add_row(pmuData[1], str(pmuData[0]),pmuData[2], str(pmuNumbers))
            else:
                table.add_row(pmuData[2], "<Not Supported>"," ", str(pmuNumbers))
    fileObject.close()


def initLogs():
    global table
    table = Table(title="PMUs Evaluation")
    table.add_column("PMU Name", style="cyan", no_wrap=True)
    table.add_column("# Events", justify="right", style="magenta")
    table.add_column("% Time", style="yellow")
    table.add_column("# Run", justify="right", style="green")

    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    fileObject = open(PERF_REPORT_FILENAME, 'w')
    fileObject.write("--------------")
    fileObject.write("START BENCHING: " + dt_string)
    fileObject.write("--------------")
    fileObject.close()

def populateTable(table):
    fileLines = open(PERF_REPORT_FILENAME, 'r').readlines()

    for line in fileLines:
        pass

    #
    # table.add_row("May 25, 2018", "Solo: A Star Wars Story", "$393,151,347")
    # table.add_row("Dec 15, 2017", "Star Wars Ep. V111: The Last Jedi", "$1,332,539,889")
    # table.add_row("Dec 16, 2016", "Rogue One: A Star Wars Story", "$1,332,439,889")
    fileObject.close()


def main():
    global table, RUN_COUNTER, args
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
                perfList.append(line.replace("\n","").split("/")[1])

        
    console.print(perfList)
    console.print("numbers of fetchable PMU events: " + str(len(perfList)))
    console.print("numbers of PMU events each run: " + str(PMU_STEPS))
    console.print("PMU events common for each run: " + str(COMMON_PMUS))

    for x in PMU_STEPS:
        pmuSelected = ",".join(perfList[0:x])
        if len(COMMON_PMUS) > 0:
            pmuSelected += "," + ",".join(COMMON_PMUS)
        # console.print(pmuSelected)
        cmdBench = ["sudo","perf" if args.sudo else "perf", "stat", "-e", pmuSelected, "sysbench", "cpu", "run"]
        with console.status("doing benchmarks on " + pmuSelected.replace(",",", ") + "..."):
            console.print(cmdBench)
            results = subprocess.run(cmdBench, text=True, capture_output=True)
            if(results.returncode == 0):
                writeLogsFile(results.stderr, pmuSelected, x) #perf writes on stderr while using stat ..
            else:
                RUNS_FAILED.append(RUN_COUNTER)
        for pmuEvaluated in perfList[0:x]:
            console.print(pmuEvaluated + (" :heavy_check_mark:" if  results.returncode == 0 else " :warning:"))
        RUN_COUNTER += 1

    console.print(table)
    console.print("benchmarks done!", style="blink")
    if(len(RUNS_FAILED) > 0):
        console.print("Failed runs: " + str(RUNS_FAILED), style="red")
    console.print("check report file in " + PERF_REPORT_FILENAME)

if __name__ == "__main__":
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                       '--sudo',
                       dest="sudo",
                       action='store_true',
                       default=False,
                       help='exec command with sudo capabilities')

    # parser.add_argument('-v',
    #                    '--verbose',
    #                     dest="verbose",
    #                     action='store_true',
    #                     help='debug prints')
    args = parser.parse_args()
    main()
 
