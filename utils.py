import numpy as np  # type: ignore
import machine, job, sol
from typing import Any, Tuple
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
def searchfile (jobs : int, machines : int, path : str = "./test/") -> str:
    """
    Given a number of jobs and a number of machines that define
    the complexity of a problem, this method looks for the file
    with the correct benchmarks.

    If the exactly specified number of jobs and machine is not found,
    an exception is raised.

    """
    try:
        filename = f"t_j{jobs}_m{machines}.txt"
        f = open(path + filename)
        f.close()
        return filename
    except:
        raise Exception("Bechmarks with required caracteristics not found.")

class Problem (object):

    """
    An instance of this class represents a problem to solve.
    """

    def __init__(self,
                jobs : int,
                machines : int,
                seed : int,
                upperbound : int,
                lowerbound : int,
                processing_times : Any  # np.array
                ) -> None:
        self.jobs = jobs
        self.machines = machines
        self.seed = seed
        self.upperbound = upperbound
        self.lowerbound = lowerbound
        self.processing_times = processing_times


    def __hash__ (self) -> str:
        return hash(str(self))

    def __repr__(self) -> str:
        return f"""
        ----------------------------------------------------------
        Jobs : {self.jobs},
        Machines : {self.machines},
        Seed : {self.seed},
        Upperbound : {self.upperbound},
        Lowerbound : {self.lowerbound},
        ProcessingTimes:
        {self.processing_times}
        ----------------------------------------------------------
        """

def readfile (filename : str, path : str = "./test/") -> Tuple[Problem]:
    """
    This method reads a file containing some Taillard's benchmarks
    and returns a set of standardized Problem instances (see class above).

    """
    # Init problem variables
    jobs, machines, seed, upper, lower = 0, 0, 0, 0, 0
    proc_times = []
    reading_general_info, reading_proc_times, counter = False, False, 0

    # Init standard headers reported in bechmark files
    signal_new_problem = "number of jobs, number of machines, initial seed, upper bound and lower bound :".strip()
    signal_proc_times = "processing times :".strip()

    # Init problems list
    probs = list()

    with open(f"{path}{filename}", 'r') as file:
        for line in file:
            cline = line.strip()

            # If next line contains general info of a new problem
            if cline == signal_new_problem:
                reading_general_info = True
                continue

            if reading_general_info:
                jobs, machines, seed, upper, lower = tuple(map(int, line.split()))
                reading_general_info = False
                continue

            # If starting from the next line, the machines processing times
            # are reported...
            if cline == signal_proc_times:
                reading_proc_times, counter = True, 0
                continue

            # If still reading the machines processing times...
            if reading_proc_times and counter < machines:
                # Save the processing times for a new machine
                proc_times.append(list(map(int, line.split())))
                counter += 1

            if reading_proc_times and counter == machines:
                reading_proc_times = False
                probs.append(Problem(jobs, machines, seed, upper, lower, processing_times=np.asarray(proc_times)))
                proc_times = []

    return tuple(probs)

def define_problem(probs, problem, beta=0.2, max_time=15, h=0.25, seed = 0, ML_method = None):

    n_machines = probs[problem].machines
    machines = []
    for i in range(n_machines):
        machines.append(machine.machine(i))

    n_jobs = probs[problem].jobs
    jobs = []
    for i in range(n_jobs):
        jobs.append(job.job(i, n_machines, times = probs[problem].processing_times.transpose()[i], n_jobs= n_jobs, ML_method = ML_method))

    solution = sol.sol(jobs, machines, beta=beta, max_time=max_time, h=h, seed = seed, ML_method = ML_method)
    #solution.multistart_NEH()
    return solution

def read_data(text):
    with open(text) as f:
        problem = ""
        for i in f:
            if i[0] == "#":
                continue
            else:
                text = i.split(";")
                probs = readfile(text[0], path="./test/")

                if problem != text[0]:
                    problem = text[0]
                    data = pd.read_table("data_base.txt", sep=" ")
                    data = data[data.h == float(text[2])]
                    y = data["time_black_box"]
                    x = data.loc[:, ["Machine_processed_jobs", "time_of_job_in_machine", "job_class"]]
                    rf = RandomForestRegressor(bootstrap=True, max_depth=20, max_features=3, min_samples_leaf=4,
                                               min_samples_split=3, n_estimators=150, random_state=0)
                    rf.fit(x.values, y.values)

                    solution = define_problem(probs, 0, beta=float(text[1]), h=float(text[2]), max_time=float(text[3]),
                                              seed=int(text[4]), ML_method=rf)

                solution.beta = float(text[1])
                solution.solution = []
                solution.makespan = 0
                solution.h = float(text[2])
                np.random.seed(int(text[4]))
                #rf = pickle.load(open(text[4], 'rb'))

                #Determinista
                if int(text[5]) == 1:
                    solution.multistart_NEH_acceleration()
                elif int(text[5]) == 2:
                    solution.multistart_NEH_acceleration_LS1()
                elif int(text[5]) == 3:
                    solution.multistart_NEH_acceleration_LS_largo()
                elif int(text[5]) == 4:
                    solution.multistart_NEH_acceleration_LS_profundo()
                elif int(text[5]) == 5:
                    solution.multistart_NEH_acceleration_LS_best()

                #Solo el NEH
                elif int(text[5]) == 6:
                    solution.multistart_NEH_acceleration_NEH_only()

                #ML
                elif int(text[5]) == 7:
                    solution.multistart_NEH_ML()
                elif int(text[5]) == 8:
                    solution.multistart_NEH_ML_largo()
                elif int(text[5]) == 9:
                    solution.multistart_NEH_ML_profundo()

                #Solo el NEH
                elif int(text[5]) == 10:
                    solution.multistart_NEH_ML_NEH_only()

                print()

                write_solution(solution, text)

def write_solution(sol, text):
    if os.path.exists('Results3.txt'):
        with open('Results3.txt', 'a') as f:
            #       instance_name             beta                  h                max_time                seed             true/false               makespan_deterministic                       makespan
            f.write(str(text[0]) + ";" + str(text[1]) + ";" + str(text[2]) + ";" + str(text[3]) + ";" + str(text[4]) + ";" + str(text[5]) + ";" + str(sol.deterministic_makespan()) + ";" + str(sol.bb_makespan()) + "\n")
    else:
        with open('Results3.txt', 'a') as f:
            f.write("instance_name" + ";" + "beta" + ";" + "h" + ";" + "max_time" + ";" + "seed" + ";" + "ML_used" + ";" + "makespan" + ";" + "makespan_bb" + "\n")
            f.write(str(text[0]) + ";" + str(text[1]) + ";" + str(text[2]) + ";" + str(text[3]) + ";" + str(text[4]) + ";" + str(text[5]) + ";" + str(sol.deterministic_makespan()) + ";" + str(sol.bb_makespan()) + "\n")

    f.close()
def create_run_file():
    #	Instance	beta	h	max_time    ML_Method   seed
    files = ["t_j20_m5.txt", "t_j20_m10.txt", "t_j20_m20.txt", "t_j50_m5.txt", "t_j50_m10.txt", "t_j50_m20.txt",
             "t_j100_m5.txt", "t_j100_m10.txt", "t_j100_m20.txt", "t_j200_m10.txt", "t_j200_m20.txt", "t_j500_m20.txt"]
    #files = ["t_j100_m5.txt", "t_j100_m10.txt", "t_j100_m20.txt", "t_j200_m10.txt", "t_j200_m20.txt", "t_j500_m20.txt"]
    #files = ["t_j200_m20.txt"]
    beta = [0.4]
    h = [0.75]
    max_time = [60, 180]
    seeds = [12345, 435451, 56162, 68813, 368387, 687619,354135,839113,123452,4354512]  # Para determinar LS
    #
    #seeds =[12345,435451,54268,68813,368387,687619,354135,839113,123452,4354512,542682,688132,3683872,6876192,3541352,8391132] Para determinar el Beta

    ML_method = [7,8,9]
    if not os.path.isfile('test.txt'):
        f = open("test.txt", "a")
        f.write("#	Instance	beta	h	max_time    ML_Method   seed   ML\n")

    with open('test.txt', 'a') as f:
        for file in files:
            for b in beta:
                for i in h:
                    for seed in seeds:
                        for ML in ML_method:
                            if int(file.split("_")[1][1:]) in [100, 200, 500]:
                                f.write(str(file) + ";" + str(b) + ";" + str(i) + ";" + str(max_time[1]) + ";" + str(
                                    seed) + ";" + str(ML) + ";" + "\n")
                            else:
                                f.write(str(file) + ";" + str(b) + ";" + str(i) + ";" + str(max_time[0]) + ";" + str(
                                    seed) + ";" + str(ML) + ";" + "\n")




if __name__ == "__main__":
    """
    Use example.
    
    """

    files = ["t_j20_m5.txt", "t_j20_m10.txt", "t_j20_m20.txt", "t_j50_m5.txt", "t_j50_m10.txt", "t_j50_m20.txt",
             "t_j100_m5.txt", "t_j100_m10.txt", "t_j100_m20.txt", "t_j200_m10.txt", "t_j200_m20.txt", "t_j500_m20.txt"]

    h_var = [2.5, 5, 7.5]
    """ 
    probs = readfile("t_j20_m5.txt", path="./test/")
    # for p in probs:
    #    print(p)

    sol = define_problem(probs, 0)  # 0 represents the problem that we are solving and probs the group of problems that we have.
    sol.build_dataset(10)
    """
    for f in files:
        for h in h_var:
            probs = readfile(f, path="./test/")
            #for p in probs:
            #    print(p)

            solution = define_problem(probs, 0, h=h, max_time=5) # 0 represents the problem that we are solving and probs the group of problems that we have.
            solution.build_dataset(10)
