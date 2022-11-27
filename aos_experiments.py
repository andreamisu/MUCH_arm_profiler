#make collect-tests-profile
#make profile
#make check-test-alarm

import subprocess
from ruamel.yaml import YAML
from rich.console import Console
from collections import defaultdict

AOS_CONFIGURATION_YML_PATH = 'original/parser/configurations.yaml'
PMU_SUPPORTED = []
PMU_STEPS = 5
PMU_RESULTS = defaultdict(list)
MUCH_EXECUTED_ITERATION = defaultdict(list)
EFFICIENT_PMUS_ALLOCATION = 6
EXPERIMENTS_LIST = []
EXPERIMENTS_ITERATION = 400 

def checkComplementaryPMUIterations(chosenMuchPmus, secondaryPMU):
    global MUCH_EXECUTED_ITERATION
    for pmu in chosenMuchPmus:
        try:
            if secondaryPMU in MUCH_EXECUTED_ITERATION[pmu]:
                return False
        except KeyError:
            MUCH_EXECUTED_ITERATION[pmu] = []
    return True

def main():
    global PMU_SUPPORTED, MUCH_EXECUTED_ITERATION

    console = Console()

    #INTRO
    #remove kernel module
    ok = subprocess.run(['make', 'remove_kernel_module'], text=True, capture_output=True)
    print(ok.stdout)
    print(ok.stderr)
    #clear dmesg
    ok = subprocess.run(['sudo' ,'dmesg', '-c'], text=True, capture_output=True)
    print(ok.stdout)
    print(ok.stderr)
    ok = subprocess.run(['sudo' ,'insmod', 'monitoring.ko'], text=True, capture_output=True)
    print(ok.stdout)
    print(ok.stderr)
    #FETCH Supported Counters
    dmesg = subprocess.run(['dmesg'], text=True, capture_output=True)
    print(dmesg.stdout)
    if(dmesg.returncode == 0):
        for line in dmesg.stdout.splitlines():
            if 'Supported counters' in line:
                PMU_SUPPORTED = line.split(':')[1].split(" ")[1:-1]

    if len(PMU_SUPPORTED) == 0:
        print("no PMU were fetched by dmesg, shutting down.")
        return -1

    yaml = YAML()
    with open(AOS_CONFIGURATION_YML_PATH, 'r') as f:
        config = yaml.load(f)

    bands = []
    for pmu in PMU_SUPPORTED:
        band = {
            'bands': [999999999],
            'description': 'Read ' + pmu + ' PMU for this resource',
            'filter': 'VU_r1',
            'resource': pmu,
            'source': ['PMU', pmu]
        }
        bands.append(band)
    config["resource_bands"] = bands
    with open(AOS_CONFIGURATION_YML_PATH, 'w') as f:
        yaml.dump(config, f)

    for pivotPMU in PMU_SUPPORTED:
        while len(set(MUCH_EXECUTED_ITERATION[pivotPMU])) < len(PMU_SUPPORTED):
            chosenMuchPmus = [pivotPMU]
            #TODO: Shall we take into account also multiple iteration on tuple of PMUs repeteing?
            for secondaryPMU in list(filter(lambda elm: elm not in MUCH_EXECUTED_ITERATION[pivotPMU], PMU_SUPPORTED)):
                if secondaryPMU == pivotPMU:
                    continue
                if checkComplementaryPMUIterations(chosenMuchPmus, secondaryPMU):
                    chosenMuchPmus.append(secondaryPMU)
                    if len(chosenMuchPmus) == EFFICIENT_PMUS_ALLOCATION:
                        break #reached max allocable pmus

            for x in chosenMuchPmus:
                MUCH_EXECUTED_ITERATION[x].extend(chosenMuchPmus)
            EXPERIMENTS_LIST.append(chosenMuchPmus)
    print("experiments: \n%s" % (str(EXPERIMENTS_LIST)))


    for elm in EXPERIMENTS_LIST:
        with open(AOS_CONFIGURATION_YML_PATH, 'r') as f:
            exp = yaml.load(f)
        # nullstring = b'null'
        experiment = [{
            'conditions': None,
            'name': 'shared',
            'resources': elm,
            'strategy': "TaskB"
        }]
        exp['task_profiles'] = experiment
        with open(AOS_CONFIGURATION_YML_PATH, 'w') as f:
            yaml.dump(exp, f)

        with open(AOS_CONFIGURATION_YML_PATH, 'r') as f:
            config = yaml.load(f)
            print(exp['task_profiles'])
    
        with console.status("make ...", spinner="monkey"):
            make = subprocess.run(['make'], text=True, capture_output=True)
            print(make.stdout)
        #collect test profile
        with console.status("make collect-tests-profile...", spinner="monkey"):
            collect = subprocess.run(['make', 'collect-tests-profile'], text=True, capture_output=True)
            print(collect.stdout)

        #make profile
        with console.status("make profile...", spinner="monkey"):
            profile = subprocess.run(['make', 'profile'], text=True, capture_output=True)
            print(profile.stdout)

        with open(AOS_CONFIGURATION_YML_PATH, 'r') as f:
            config = yaml.load(f)
            print(exp['task_profiles'])
        
        for n in range(EXPERIMENTS_ITERATION):
            #make check-test-alarm
            with console.status("EXP "+ str(n) +" :: make check-tests-alarm...", spinner="monkey"):
                test = subprocess.run(['make', 'check-tests-alarm'], text=True, capture_output=True)
                print(test.stdout)

                #FETCH Events
                dmesg = subprocess.run(['dmesg'], text=True, capture_output=True)
                print(dmesg.stdout)
                if(dmesg.returncode == 0):
                    for line in dmesg.stdout.splitlines():
                        if 'monitoring' in line and 'counter' in line and '(' in line:
                            eventName = line.split('(')[1].split(")")[0].rstrip()
                            eventNumber = line.split(':')[2].rstrip()
                            print("fetched %s : %s" %(eventName , eventNumber))
                            PMU_RESULTS[eventName].append(eventNumber)

        for pmu in elm:
            console.print(PMU_RESULTS[pmu])


    
if __name__ == "__main__":
    main()
