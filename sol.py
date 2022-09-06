import time
import copy
import operator
from machine import machine
from math import log
import numpy as np
import random
import os
from ML import train_random_forest

class sol:
    def __init__(self, jobs, machines, beta = 0.99, max_iter = 3, max_time = 15, h = 0.5, seed = 0, ML_method = None):
        """
        We are going to initialize our solution with the following variables:

        :param jobs: The jobs that we have, with its processing times and black box
        :param machines: Machines that we have.
        :param beta: Parameter for our BR-NEH
        :param max_time: Time that is going to be spend for solving our algorithm
        :param h: Black box parameter
        """
        self.jobs = jobs
        self.machines = machines
        self.solution = []
        self.beta = beta
        self.max_iter = max_iter #Local search largo
        self.max_time = max_time
        self.h = h
        self.makespan = 0
        self.ML_method = ML_method

        np.random.seed(seed)

    def copy_solution(self):
        sol = []
        for i in self.solution:
            sol.append(i)
        return sol

    def seleccionar(self, lista):
        """
        To select the next element that is going to enter
        :param lista: Our list sorted by decresing order (if we don't reorder it with our bias function)
        :return: The element that is going to be included in our solution
        """
        return lista.pop(0)

    def get_position(self, n: int) -> int:
        """
        To get the next position based on a beta distribution
        :param n: The number of element of our list. We put it just to don't put a number that is outside of our list.
        :return: A random number from a beta distribution
        """
        return int((log(np.random.random()) / log(1 - self.beta))) % n

    def reorganize_list(self, v):
        """
        We reorganize our list, to have different solution each time we rerun our algorithm
        :param v: The list that we need to reorganize
        :return: Our reorganize list
        """
        monte_carlo = []
        lista = v[:]
        while lista:
            position = self.get_position(len(lista))
            monte_carlo.append(lista.pop(position))

        return monte_carlo

    def show(self, plan):
        """
        Just to show the solution
        :param plan: It's a tupla with our jobs, our machines and the makespan.
        :return:
        """
        print("The list of jobs are:", [i.id for i in plan[0]])
        print("Our machines have the next times:", [i.time for i in plan[1]])
        print("And our makespan is:", plan[2])

    def NEH(self):
        """
        1) First, we create a list with our jobs. We sort it depending of the time that each job spend until it finishes.
        2) We reorganize this list using BR.
        3) We use NEH to complete our solution
        4) We return our makespan
        """
        #1
        lista = [(i, i.total_time, j) for j,i in enumerate(self.jobs)]
        lista.sort(reverse=True, key= lambda x: x[1])

        #2
        lista = self.reorganize_list(lista)

        #3
        self.solution.append(lista.pop(0)[0])
        for _ in range(len(lista)):
            self.insert(lista)

        #4
        return [i.time for i in self.machines][-1]

    def NEH_with_ML_method(self):
        """
        1) First, we create a list with our jobs. We sort it depending of the time that each job spend until it finishes.
        2) We reorganize this list using BR.
        3) We use NEH to complete our solution
        4) We return our makespan
        """
        #1
        lista = [(i, i.total_time, j) for j,i in enumerate(self.jobs)]
        lista.sort(reverse=True, key= lambda x: x[1])

        #2
        lista = self.reorganize_list(lista)

        #3
        self.solution.append(lista.pop(0)[0])
        length = len(lista)
        for _ in range(length):
            self.insert_ML(lista)

        #4
        return [i.time for i in self.machines][-1]

    def NEH_acceleration(self):
        """
        1) First, we create a list with our jobs. We sort it depending of the time that each job spends until it finishes.
        2) We reorganize this list using BR.
        3) We use NEH to complete our solution
        4) We return our makespan
        """
        #1
        lista = [(i, i.total_time, j) for j,i in enumerate(self.jobs)]
        lista.sort(reverse=True, key= lambda x: x[1])

        #2
        lista = self.reorganize_list(lista)

        #3
        self.solution.append(lista.pop(0)[0])
        for _ in range(len(lista)):
            best, position = self.taylor_acceleration(lista[0][0])
            self.solution.insert(position, lista.pop(0)[0])

        #4
        return best

    def taylor_acceleration(self, job):
        k = len(self.solution)+1
        m = len(self.machines)

        e = np.matrix([[0, ] * (m+1), ] * k)
        q = np.matrix([[0, ] * (m+1), ] * (k+1))
        f = np.matrix([[0, ] * (m+1), ] * (k+1))

        best = -1
        position = 0
        for i in range(1, k):
            for j in range(1, m+1):
                e[i, j] = max([e[i, j-1], e[i-1, j]]) + self.jobs[self.solution[i-1].id].time_machines[j-1]
                q[k-i, m-j] = max([q[k-i, m-j+1], q[k-i+1, m-j]]) + self.jobs[self.solution[k-i-1].id].time_machines[m-j]
                f[i, j] = max([f[i, j-1], e[i-1, j]]) + job.time_machines[j-1]
                if i == k-1:
                    f[k, j] = max([f[k, j - 1], e[k - 1, j]]) + job.time_machines[j - 1]

        for i in range(1, k):
            list = []
            for j in range(1, m+1):
                list.append(f[i, j] + q[i, j - 1])
            makespan = max(list)
            #print(makespan)
            if best == -1 or best > makespan:
                best = makespan
                position = i-1
        if best > f[k, m]:
            best = f[k, m]
            position = k

        return (best, position)

    def insert(self, lista):
        data = self.seleccionar(lista)[0]
        best = []
        for i in range(len(self.solution) + 1):
            #aux_solution = copy.deepcopy(self.solution)
            aux_solution = self.copy_solution()
            #aux_machines = copy.deepcopy(self.machines)
            aux_machines = [machine(i) for i in range(len(self.machines))]
            aux_solution = aux_solution[:i] + [data] + aux_solution[i:]

            for j in aux_solution:
                for k in aux_machines:
                    if k.id == 0:
                        k.time += j.time_machines[k.id]
                    elif aux_machines[k.id - 1].time <= k.time:
                        k.time += j.time_machines[k.id]
                    else:
                        k.time = aux_machines[k.id - 1].time + j.time_machines[k.id]

            best.append((aux_machines[-1].time, aux_machines, aux_solution))

        best.sort(key=operator.itemgetter(0))   # best.sort(key=lambda x: x[0])   
        self.solution = best[0][2]
        self.machines = best[0][1]

    def insert_ML(self, lista):
        #data_base.write(str(self.h) + " " + str(k.processed_jobs/len(self.jobs)) + " " + str(j.time_machines[k.id])  + " "  + str(time_black_box) + " " + str(j.job_class)  + " " + str(k.time) + " " + str(j.id) + " " + str(k.id) + "\n")
        # data_base.write("h" + " Machine_processed_jobs" + " time_of_job_in_machine" + " time_black_box" + " job_class"  + " machine_time" + " job_id" + " machine_id" + "\n")
        #                      k.processed_jobs/len(self.jobs)  j.time_machines[k.id]   j.job_class  k.time
        data = self.seleccionar(lista)[0]
        best = []
        for i in range(len(self.solution) + 1):
            #aux_solution = copy.deepcopy(self.solution)
            aux_solution = self.copy_solution()
            #aux_machines = copy.deepcopy(self.machines)
            aux_machines = [machine(i) for i in range(len(self.machines))]
            aux_solution = aux_solution[:i] + [data] + aux_solution[i:]

            for j in aux_solution:
                for k in aux_machines:

                    predict_time = j.ML_times[k.id][k.processed_jobs]

                    if k.id == 0:
                        k.time += predict_time
                    elif aux_machines[k.id - 1].time <= k.time:
                        k.time += predict_time
                    else:
                        k.time = aux_machines[k.id - 1].time + predict_time

                    k.processed_jobs += 1

            best.append((aux_machines[-1].time, aux_machines, aux_solution))

        best.sort(key=operator.itemgetter(0))   # best.sort(key=lambda x: x[0])
        self.solution = best[0][2]
        self.machines = best[0][1]

    def multistart_NEH(self):
        start_time = time.time()
        best = -1
        best_planning = ()
        while self.max_time > time.time() - start_time:
            makespan = self.NEH()
            if best == -1 or best > makespan:
                #best_planning = (copy.deepcopy(self.solution), copy.deepcopy(self.machines), makespan)
                best_planning = (self.copy_solution(), copy.deepcopy(self.machines), makespan)
                best = makespan
            self.solution = []
            self.machines = [machine(i) for i in range(len(self.machines))]

        self.solution = best_planning[0]
        makespan_bb = self.bb_makespan()
        self.show(best_planning)
        print("Makespan for our black box is: ", makespan_bb)

    def multistart_NEH_ML(self):
        start_time = time.time()
        best = -1
        best_planning = ()
        while self.max_time > time.time() - start_time:
            makespan = self.NEH_with_ML_method()
            #print(makespan)
            self.Local_Search_ML()
            makespan = self.deterministic_makespan()
            #print(makespan)
            if best == -1 or best > makespan:
                #best_planning = (copy.deepcopy(self.solution), copy.deepcopy(self.machines), makespan)
                best_planning = (self.copy_solution(), copy.deepcopy(self.machines), makespan)
                best = makespan
            self.solution = []
            self.machines = [machine(i) for i in range(len(self.machines))]

        self.solution = best_planning[0]
        makespan_bb = self.bb_makespan()
        self.show(best_planning)
        print("Makespan for our black box is: ", makespan_bb)
        print("Makespan deterministic: ", self.deterministic_makespan())

    def multistart_NEH_ML_largo(self):
        start_time = time.time()
        best = -1
        best_planning = ()
        while self.max_time > time.time() - start_time:
            makespan = self.NEH_with_ML_method()
            #print(makespan)
            self.Local_Search_ML_largo()
            makespan = self.deterministic_makespan()
            #print(makespan)
            if best == -1 or best > makespan:
                #best_planning = (copy.deepcopy(self.solution), copy.deepcopy(self.machines), makespan)
                best_planning = (self.copy_solution(), copy.deepcopy(self.machines), makespan)
                best = makespan
            self.solution = []
            self.machines = [machine(i) for i in range(len(self.machines))]

        self.solution = best_planning[0]
        makespan_bb = self.bb_makespan()
        self.show(best_planning)
        print("Makespan for our black box is: ", makespan_bb)
        print("Makespan deterministic: ", self.deterministic_makespan())

    def multistart_NEH_ML_profundo(self):
        start_time = time.time()
        best = -1
        best_planning = ()
        while self.max_time > time.time() - start_time:
            makespan = self.NEH_with_ML_method()
            #print(makespan)
            self.Local_Search_ML_profundo()
            makespan = self.deterministic_makespan()
            #print(makespan)
            print()
            if best == -1 or best > makespan:
                #best_planning = (copy.deepcopy(self.solution), copy.deepcopy(self.machines), makespan)
                best_planning = (self.copy_solution(), copy.deepcopy(self.machines), makespan)
                best = makespan
            self.solution = []
            self.machines = [machine(i) for i in range(len(self.machines))]

        self.solution = best_planning[0]
        makespan_bb = self.bb_makespan()
        self.show(best_planning)
        print("Makespan for our black box is: ", makespan_bb)
        print("Makespan deterministic: ", self.deterministic_makespan())

    def multistart_NEH_ML_NEH_only(self):
        start_time = time.time()
        best = -1
        best_planning = ()
        while self.max_time > time.time() - start_time:
            _ = self.NEH_with_ML_method()
            makespan = self.deterministic_makespan()
            if best == -1 or best > makespan:
                #best_planning = (copy.deepcopy(self.solution), copy.deepcopy(self.machines), makespan)
                best_planning = (self.copy_solution(), copy.deepcopy(self.machines), makespan)
                best = makespan
            self.solution = []
            self.machines = [machine(i) for i in range(len(self.machines))]

        self.solution = best_planning[0]
        makespan_bb = self.bb_makespan()
        self.show(best_planning)
        print("Makespan for our black box is: ", makespan_bb)
        print("Makespan deterministic: ", self.deterministic_makespan())

    def Local_Search_ML(self):
        improve = True
        while improve:
            improve = False
            job = self.solution.index(random.choice(self.solution))#range(1, len(self.solution)-1, 2)
            best = [(self.machines[-1].time, self.machines, self.solution)]
            #best.append((aux_machines[-1].time, aux_machines, aux_solution))
            for i in range(job-1, job+2):
                if i == job:
                    continue
                # aux_solution = copy.deepcopy(self.solution)
                aux_solution = self.copy_solution()
                data = aux_solution.pop(job)
                # aux_machines = copy.deepcopy(self.machines)
                aux_machines = [machine(i) for i in range(len(self.machines))]
                aux_solution = aux_solution[:i] + [data] + aux_solution[i:]

                for j in aux_solution:
                    for k in aux_machines:
                            # predict_time = self.ML_method.predict([[k.processed_jobs/len(self.jobs),  j.time_machines[k.id],   j.job_class]])[0]
                        predict_time = j.ML_times[k.id][k.processed_jobs]
                            # print("self.ML_method.predict([[k.processed_jobs/len(self.jobs),  j.time_machines[k.id],   j.job_class]])[0]: " + str(self.ML_method.predict([[k.processed_jobs/len(self.jobs),  j.time_machines[k.id],   j.job_class]])[0]))
                            # print("j.time_machines[k.id]: " + str(j.time_machines[k.id]))
                            # print("j.black_box(self.h, k, len(self.jobs)): " + str(j.black_box(self.h, k, len(self.jobs))))
                            # predict_time = j.black_box(self.h, k, len(self.jobs))
                        if k.id == 0:
                            k.time += predict_time
                        elif aux_machines[k.id - 1].time <= k.time:
                            k.time += predict_time
                        else:
                            k.time = aux_machines[k.id - 1].time + predict_time
                        k.processed_jobs += 1
                best.append((aux_machines[-1].time, aux_machines, aux_solution))

            best_value = best[0][0]
            for i in range(2):
                if best_value > best[i+1][0]:
                    best_value = best[i + 1][0]
                    self.solution = best[i + 1][2]
                    self.machines = best[i + 1][1]
                    improve = True

    def Local_Search_ML_largo(self):
        improve = True
        count = 0
        while improve or self.max_iter >= count:
            improve = False
            count += 1
            job = self.solution.index(random.choice(self.solution)) #range(1, len(self.solution)-1, 2)
            best = [(self.machines[-1].time, self.machines, self.solution)]
            #best.append((aux_machines[-1].time, aux_machines, aux_solution))
            for i in range(job-1, job+2):
                if i == job:
                    continue
                # aux_solution = copy.deepcopy(self.solution)
                aux_solution = self.copy_solution()
                data = aux_solution.pop(job)
                # aux_machines = copy.deepcopy(self.machines)
                aux_machines = [machine(i) for i in range(len(self.machines))]
                aux_solution = aux_solution[:i] + [data] + aux_solution[i:]

                for j in aux_solution:
                    for k in aux_machines:
                            # predict_time = self.ML_method.predict([[k.processed_jobs/len(self.jobs),  j.time_machines[k.id],   j.job_class]])[0]
                        predict_time = j.ML_times[k.id][k.processed_jobs]
                            # print("self.ML_method.predict([[k.processed_jobs/len(self.jobs),  j.time_machines[k.id],   j.job_class]])[0]: " + str(self.ML_method.predict([[k.processed_jobs/len(self.jobs),  j.time_machines[k.id],   j.job_class]])[0]))
                            # print("j.time_machines[k.id]: " + str(j.time_machines[k.id]))
                            # print("j.black_box(self.h, k, len(self.jobs)): " + str(j.black_box(self.h, k, len(self.jobs))))
                            # predict_time = j.black_box(self.h, k, len(self.jobs))
                        if k.id == 0:
                            k.time += predict_time
                        elif aux_machines[k.id - 1].time <= k.time:
                            k.time += predict_time
                        else:
                            k.time = aux_machines[k.id - 1].time + predict_time
                        k.processed_jobs += 1
                best.append((aux_machines[-1].time, aux_machines, aux_solution))

            best_value = best[0][0]
            for i in range(2):
                if best_value > best[i+1][0]:
                    best_value = best[i + 1][0]
                    self.solution = best[i + 1][2]
                    self.machines = best[i + 1][1]
                    improve = True
                    count = 0

    def Local_Search_ML_profundo(self):
        improve = True
        while improve:
            improve = False
            for job in range(1, len(self.solution) - 1, 2):
                best = [(self.machines[-1].time, self.machines, self.solution)]
                # best.append((aux_machines[-1].time, aux_machines, aux_solution))
                for i in range(job - 1, job + 2):
                    if i == job:
                        continue
                    # aux_solution = copy.deepcopy(self.solution)
                    aux_solution = self.copy_solution()
                    data = aux_solution.pop(job)
                    # aux_machines = copy.deepcopy(self.machines)
                    aux_machines = [machine(i) for i in range(len(self.machines))]
                    aux_solution = aux_solution[:i] + [data] + aux_solution[i:]

                    for j in aux_solution:
                        for k in aux_machines:
                            # predict_time = self.ML_method.predict([[k.processed_jobs/len(self.jobs),  j.time_machines[k.id],   j.job_class]])[0]
                            predict_time = j.ML_times[k.id][k.processed_jobs]
                            # print("self.ML_method.predict([[k.processed_jobs/len(self.jobs),  j.time_machines[k.id],   j.job_class]])[0]: " + str(self.ML_method.predict([[k.processed_jobs/len(self.jobs),  j.time_machines[k.id],   j.job_class]])[0]))
                            # print("j.time_machines[k.id]: " + str(j.time_machines[k.id]))
                            # print("j.black_box(self.h, k, len(self.jobs)): " + str(j.black_box(self.h, k, len(self.jobs))))
                            # predict_time = j.black_box(self.h, k, len(self.jobs))
                            if k.id == 0:
                                k.time += predict_time
                            elif aux_machines[k.id - 1].time <= k.time:
                                k.time += predict_time
                            else:
                                k.time = aux_machines[k.id - 1].time + predict_time
                            k.processed_jobs += 1
                    best.append((aux_machines[-1].time, aux_machines, aux_solution))

                best_value = best[0][0]
                for i in range(2):
                    if best_value > best[i + 1][0]:
                        best_value = best[i + 1][0]
                        self.solution = best[i + 1][2]
                        self.machines = best[i + 1][1]
                        improve = True

    def multistart_NEH_acceleration(self):
        start_time = time.time()
        best = -1
        best_planning = ()
        while self.max_time > time.time() - start_time:
            self.makespan = self.NEH_acceleration()
            if best == -1 or best > self.makespan:
                best_planning = (self.copy_solution(), self.makespan)
                best = self.makespan
            self.solution = []

        self.solution = best_planning[0]
        self.makespan = best_planning[1]
        makespan_bb = self.bb_makespan()
        #self.show(best_planning)
        print("Makespan is: ", self.makespan)
        print("Makespan for our black box is: ", makespan_bb)

    def multistart_NEH_acceleration_LS1(self):
        start_time = time.time()
        best = -1
        best_planning = ()
        while self.max_time > time.time() - start_time:
            self.makespan = self.NEH_acceleration()
            self.Local_Search()
            if best == -1 or best > self.makespan:
                best_planning = (self.copy_solution(), self.makespan)
                best = self.makespan
            self.solution = []

        self.solution = best_planning[0]
        self.makespan = best_planning[1]
        makespan_bb = self.bb_makespan()
        #self.show(best_planning)
        print("Makespan is: ", self.makespan)
        print("Makespan for our black box is: ", makespan_bb)

    def multistart_NEH_acceleration_LS_largo(self):
        start_time = time.time()
        best = -1
        best_planning = ()
        while self.max_time > time.time() - start_time:
            self.makespan = self.NEH_acceleration()
            self.Local_Search_largo()
            if best == -1 or best > self.makespan:
                best_planning = (self.copy_solution(), self.makespan)
                best = self.makespan
            self.solution = []

        self.solution = best_planning[0]
        self.makespan = best_planning[1]
        makespan_bb = self.bb_makespan()
        #self.show(best_planning)
        print("Makespan is: ", self.makespan)
        print("Makespan for our black box is: ", makespan_bb)

    def multistart_NEH_acceleration_LS_profundo(self):
        start_time = time.time()
        best = -1
        best_planning = ()
        while self.max_time > time.time() - start_time:
            self.makespan = self.NEH_acceleration()
            self.Local_Search_profundo_determinsita(start_time)
            if best == -1 or best > self.makespan:
                best_planning = (self.copy_solution(), self.makespan)
                best = self.makespan
            self.solution = []

        self.solution = best_planning[0]
        self.makespan = best_planning[1]
        makespan_bb = self.bb_makespan()
        #self.show(best_planning)
        print("Makespan is: ", self.makespan)
        print("Makespan for our black box is: ", makespan_bb)

    def multistart_NEH_acceleration_LS_best(self):
        start_time = time.time()
        best = -1
        best_planning = ()
        while self.max_time > time.time() - start_time:
            self.makespan = self.NEH_acceleration()
            self.Local_Search_best_determinsita(start_time)
            if best == -1 or best > self.makespan:
                best_planning = (self.copy_solution(), self.makespan)
                best = self.makespan
            self.solution = []

        self.solution = best_planning[0]
        self.makespan = best_planning[1]
        makespan_bb = self.bb_makespan()
        #self.show(best_planning)
        print("Makespan is: ", self.makespan)
        print("Makespan for our black box is: ", makespan_bb)


    def multistart_NEH_acceleration_NEH_only(self):
        start_time = time.time()
        best = -1
        best_planning = ()
        while self.max_time > time.time() - start_time:
            self.makespan = self.NEH_acceleration()
            if best == -1 or best > self.makespan:
                best_planning = (self.copy_solution(), self.makespan)
                best = self.makespan
            self.solution = []

        self.solution = best_planning[0]
        self.makespan = best_planning[1]
        makespan_bb = self.bb_makespan()
        #self.show(best_planning)
        print("Makespan is: ", self.makespan)
        print("Makespan for our black box is: ", makespan_bb)

    def bb_makespan(self, save = False):
        #aux_solution = copy.deepcopy(self.solution)
        aux_solution = self.copy_solution()
        if save:
            data_base = open("data_base.txt","a")
            if os.stat("data_base.txt").st_size == 0:
                data_base.write("h" + " Machine_processed_jobs" + " time_of_job_in_machine" + " time_black_box" + " job_class"  + " machine_time" + " job_id" + " machine_id" + "\n")
        aux_machines = [machine(i) for i in range(len(self.machines))]
        n = len(self.jobs)
        for j in aux_solution:
            for k in aux_machines:
                time_black_box = j.black_box(self.h, k, n)
                if k.id == 0:
                    k.time += time_black_box
                elif aux_machines[k.id - 1].time <= k.time:
                    k.time += time_black_box
                else:
                    k.time = aux_machines[k.id - 1].time + time_black_box

                if save:
                    data_base.write(str(self.h) + " " + str(k.processed_jobs/len(self.jobs)) + " " + str(j.time_machines[k.id])  + " "  + str(time_black_box) + " " + str(j.job_class)  + " " + str(k.time) + " " + str(j.id) + " " + str(k.id) + "\n")
                k.processed_jobs += 1
        if save:
            data_base.close()

        return aux_machines[-1].time

    def deterministic_makespan(self):

        aux_solution = self.copy_solution()

        aux_machines = [machine(i) for i in range(len(self.machines))]

        for j in aux_solution:
            for k in aux_machines:
                if k.id == 0:
                    k.time += j.time_machines[k.id]
                elif aux_machines[k.id - 1].time <= k.time:
                    k.time += j.time_machines[k.id]
                else:
                    k.time = aux_machines[k.id - 1].time + j.time_machines[k.id]

        return aux_machines[-1].time

    def build_dataset(self, n_solutions):
        """
        We build "n_solutions" solutions and we introduce the data that we have obtained in our data_base.txt.
        We will use it for doing our learnheuristic approach.
        :param n_solutions: Number of solutions that we want in our data base
        """
        self.beta = 0.01
        #copy_machines = copy.deepcopy(self.machines)
        copy_machines = [machine(i) for i in range(len(self.machines))]
        for _ in range(n_solutions):
            self.NEH_acceleration()
            self.bb_makespan(save=True)
            self.solution = []
            self.machines = copy_machines

    def Local_Search(self):
        improve = True
        last_used = -1
        while improve:
            improve = False
            list = [i for i in self.solution if i.id != last_used]

            used = list[random.randint(0, len(list)-1)]
            last_used = used.id
            old_pos = self.solution.index(used)
            self.solution.remove(used)
            best, position = self.taylor_acceleration(used)

            if best < self.makespan:
                improve = True
                self.solution.insert(position, used)
                self.makespan = best
            else:
                self.solution.insert(old_pos, used)

    def Local_Search_largo(self):
        improve = True
        last_used = -1
        count = 0
        while improve and count < self.max_iter:
            improve = False
            list = [i for i in self.solution if i.id != last_used]

            used = list[random.randint(0, len(list)-1)]
            last_used = used.id
            old_pos = self.solution.index(used)
            self.solution.remove(used)
            best, position = self.taylor_acceleration(used)

            if best < self.makespan:
                improve = True
                count = 0
                self.solution.insert(position, used)
                self.makespan = best
            else:
                self.solution.insert(old_pos, used)

    def Local_Search_profundo_determinsita(self, start_time):
        improve = True
        while improve and self.max_time > time.time() - start_time:
            improve = False
            for i in self.solution:
                old_pos = self.solution.index(i)
                self.solution.remove(i)
                best, position = self.taylor_acceleration(i)

                if best < self.makespan:
                    improve = True
                    self.solution.insert(position, i)
                    self.makespan = best
                else:
                    self.solution.insert(old_pos, i)

    def Local_Search_best_determinsita(self, start_time):
        improve = True
        while improve and self.max_time > time.time() - start_time:
            improve = False
            best_makespan = self.makespan
            for i in self.solution:
                old_pos = self.solution.index(i)
                self.solution.remove(i)
                best, position = self.taylor_acceleration(i)

                if best < best_makespan:
                    best_sol = copy.deepcopy(self)
                    best_sol.solution.insert(position, i)
                    self.solution.insert(old_pos, i)
                    best_makespan = best
                else:
                    self.solution.insert(old_pos, i)

            if best_makespan < self.makespan:
                improve = True
                self = best_sol

    def comprobar_makespan(self):
        aux_solution = self.copy_solution()
        aux_machines = [machine(i) for i in range(len(self.machines))]
        n = len(self.jobs)
        for j in aux_solution:
            for k in aux_machines:
                time_black_box = j.black_box(0, k, n)
                if k.id == 0:
                    k.time += time_black_box
                elif aux_machines[k.id - 1].time <= k.time:
                    k.time += time_black_box
                else:
                    k.time = aux_machines[k.id - 1].time + time_black_box

                k.processed_jobs += 1

        return aux_machines[-1].time



