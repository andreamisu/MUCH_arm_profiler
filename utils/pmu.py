class PMU:
## name
## experiments: in which PMU has been allocated
    experiments = []
    events = []
    u = 0
    o = 0
    o2 = 0      
    def __init__(self, name):
        self.name = name
