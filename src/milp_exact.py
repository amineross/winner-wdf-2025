import pulp
import numpy as np
import pandas as pd
import json
import ast
import networkx as nx
import matplotlib.pyplot as plt
import sys
import time
from typing import Optional, List, Tuple, Dict, Any
from script import Instance


class ExactMILPSolver:
    """
    Solveur exact MILP pour le problème de planification des opérateurs.

    Ce solveur utilise une modélisation complète en programmation linéaire entière
    pour maximiser la production totale, tout en respectant l’ensemble des contraintes
    liées aux opérateurs, aux tâches, et aux flux instantanés.

    Auteurs:
        Amine ROSTANE, Gaël GARNIER, Augustin LOGEAIS.
    """

    def __init__(self, instance, time_limit=None, mip_gap=0.01, verbose=True):
        """Initialise le solveur MILP avec les paramètres de l’instance.

        Args:
            instance: Instance du problème à résoudre.
            time_limit (int, optionnel): Temps maximal de résolution en secondes.
            mip_gap (float): Tolérance du gap relatif MIP (ex: 0.01 = 1%).
            verbose (bool): Active ou non l'affichage des logs de progression.
        """

        self.instance = instance
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.verbose = verbose

        self.model = pulp.LpProblem("Exact_Operator_Scheduling", pulp.LpMaximize)

    def build_model(self):
        """
        Construit le modèle MILP complet avec toutes les variables et contraintes.

        Cette méthode génère les variables de décision, la fonction objectif
        (maximisation de la production finale), et les contraintes de validité du planning.
        """

        if self.verbose:
            print("Building MILP model...")
            start_time = time.time()

        self.x = {}
        for o in range(self.instance.n_operators):
            for t in range(self.instance.time_slots):
                for m in range(self.instance.n_tasks):
                    self.x[o, t, m] = pulp.LpVariable(f"x_{o}_{t}_{m}", cat=pulp.LpBinary)

        self.y = {}
        for o in range(self.instance.n_operators):
            for t in range(self.instance.time_slots):
                self.y[o, t] = pulp.LpVariable(f"y_{o}_{t}", cat=pulp.LpBinary)

        self.z = {}
        for o in range(self.instance.n_operators):
            for t in range(self.instance.time_slots):
                for m in range(self.instance.n_tasks):
                    self.z[o, t, m] = pulp.LpVariable(f"z_{o}_{t}_{m}", cat=pulp.LpBinary)

        self.task_production = {}
        for m in range(self.instance.n_tasks):
            for t in range(self.instance.time_slots):
                self.task_production[m, t] = pulp.LpVariable(f"prod_{m}_{t}", lowBound=0)

        self.task_capacity_t = {}
        for m in range(self.instance.n_tasks):
            for t in range(self.instance.time_slots):
                self.task_capacity_t[m, t] = pulp.LpVariable(f"cap_{m}_{t}", lowBound=0)

        self.leftover = {}
        for m in range(self.instance.n_tasks):
            for t in range(self.instance.time_slots + 1):
                self.leftover[m, t] = pulp.LpVariable(f"leftover_{m}_{t}", lowBound=0)

        self.is_in_session = {}
        for o in range(self.instance.n_operators):
            for t in range(self.instance.time_slots):
                for m in range(self.instance.n_tasks):
                    for s in range(max(0, t - self.instance.max_session_slots + 1), t + 1):
                        self.is_in_session[o, t, m, s] = pulp.LpVariable(
                            f"sess_{o}_{t}_{m}_{s}", cat=pulp.LpBinary
                        )

        last_task = self.instance.n_tasks - 1
        self.model += pulp.lpSum(
            self.task_production[last_task, t] for t in range(self.instance.time_slots)
        ), "MaximizeFinalProduction"

        for o in range(self.instance.n_operators):
            for t in range(self.instance.time_slots):

                self.model += (pulp.lpSum(self.x[o, t, m] 
                                          for m in range(self.instance.n_tasks))
                               + self.y[o, t] == 1)

        for o in range(self.instance.n_operators):
            start_avail = self.instance.operators_availability[o]['start']
            end_avail = self.instance.operators_availability[o]['end']
            for t in range(self.instance.time_slots):

                if t < start_avail or t > end_avail:
                    self.model += self.y[o, t] == 1

        for (task_incomp, operator_incomp) in self.instance.incompatibilities:
            for t in range(self.instance.time_slots):
                self.model += self.x[operator_incomp, t, task_incomp] == 0

        for m in range(self.instance.n_tasks):
            base_cap = self.instance.task_capacity[m]
            for t in range(self.instance.time_slots):
                self.model += self.task_capacity_t[m, t] == pulp.lpSum(
                    self.x[o, t, m] * self.instance.operators_prod_coef[o][m]
                    for o in range(self.instance.n_operators)
                ) * base_cap

        for o in range(self.instance.n_operators):
            for m in range(self.instance.n_tasks):
                for t in range(self.instance.time_slots):
                    if t == 0:
                        self.model += self.z[o, t, m] == self.x[o, t, m]
                    else:
                        self.model += self.z[o, t, m] >= self.x[o, t, m] - self.x[o, t-1, m]
                        self.model += self.z[o, t, m] <= self.x[o, t, m]
                        self.model += self.z[o, t, m] <= 1 - self.x[o, t-1, m]

        for o in range(self.instance.n_operators):
            for m in range(self.instance.n_tasks):
                for t in range(self.instance.time_slots):

                    self.model += pulp.lpSum(
                        self.is_in_session[o, t, m, s]
                        for s in range(max(0, t - self.instance.max_session_slots + 1), t + 1)
                    ) == self.x[o, t, m]

                    for s in range(max(0, t - self.instance.max_session_slots + 1), t + 1):
                        self.model += self.is_in_session[o, t, m, s] <= self.x[o, t, m]
                        self.model += self.is_in_session[o, t, m, s] <= self.z[o, s, m]
                        self.model += self.is_in_session[o, t, m, s] >= self.x[o, t, m] + self.z[o, s, m] - 1

        min_slots = self.instance.min_session_slots

        for o in range(self.instance.n_operators):
            for m in range(self.instance.n_tasks):
                for t in range(self.instance.time_slots - min_slots + 1):

                    for i in range(1, min_slots):
                        self.model += self.x[o, t + i, m] >= self.z[o, t, m]

                for t2 in range(self.instance.time_slots - min_slots + 1, self.instance.time_slots):
                    self.model += self.z[o, t2, m] == 0

        max_slots = self.instance.max_session_slots
        for o in range(self.instance.n_operators):
            for m in range(self.instance.n_tasks):

                for t in range(self.instance.time_slots - max_slots):
                    self.model += pulp.lpSum(
                        self.x[o, t + i, m] for i in range(max_slots + 1)
                    ) <= max_slots

        for o in range(self.instance.n_operators):
            for m in range(self.instance.n_tasks):
                self.model += pulp.lpSum(
                    self.x[o, t, m] for t in range(self.instance.time_slots)
                ) <= self.instance.max_task_slots

        for o in range(self.instance.n_operators):
            self.model += pulp.lpSum(
                self.x[o, t, m] for m in range(self.instance.n_tasks)
                for t in range(self.instance.time_slots)
            ) >= self.instance.min_work_slots

        bigM = 1000000  

        for m in range(self.instance.n_tasks):
            if m == 0:

                self.model += self.leftover[m, 0] == bigM
            else:

                self.model += self.leftover[m, 0] == 0

        for t in range(self.instance.time_slots):
            for m in range(self.instance.n_tasks):

                if m == 0:

                    supply_expr = self.leftover[0, t]
                else:
                    preds = list(self.instance.task_graph.predecessors(m))

                    supply_expr = self.leftover[m, t] + pulp.lpSum(
                        self.task_production[pred, t] * self.instance.task_graph[pred][m]['weight']
                        for pred in preds
                    )

                self.model += self.task_production[m, t] <= self.task_capacity_t[m, t]

                self.model += self.task_production[m, t] <= supply_expr

                if t < self.instance.time_slots:
                    if m == 0:

                        self.model += self.leftover[0, t+1] == self.leftover[0, t] - self.task_production[0, t]
                    else:
                        preds = list(self.instance.task_graph.predecessors(m))
                        self.model += (
                            self.leftover[m, t+1] 
                            == 
                            self.leftover[m, t] 
                            + pulp.lpSum(
                                self.task_production[pred, t] * self.instance.task_graph[pred][m]['weight']
                                for pred in preds
                            )
                            - self.task_production[m, t]
                        )

        if self.verbose:
            n_vars = len(self.model.variables())
            n_constraints = len(self.model.constraints)
            build_time = time.time() - start_time
            print(f"Model built with {n_vars} variables and {n_constraints} constraints in {build_time:.2f} seconds")

    def solve(self):
        """
        Résout le modèle MILP et extrait le planning optimal trouvé.

        Returns:
            tuple: Un tuple (schedule, score) où :
                - schedule (List[List[int]]): Matrice O*T représentant le planning.
                - score (float): Nombre total d’unités produites validé par la simulation.
        """

        if self.verbose:
            print(f"Solving MILP model with time limit: {self.time_limit} seconds...")
            start_time = time.time()

        if self.time_limit is not None:
            solver = pulp.PULP_CBC_CMD(
                timeLimit=self.time_limit,
                gapRel=self.mip_gap,
                msg=self.verbose
            )
        else:
            solver = pulp.PULP_CBC_CMD(
                gapRel=self.mip_gap,
                msg=self.verbose
            )

        self.model.solve(solver)
        solve_time = time.time() - start_time
        status = pulp.LpStatus[self.model.status]

        if self.verbose:
            print(f"Solving completed in {solve_time:.2f} seconds with status: {status}")

        if status not in ["Optimal", "Feasible"]:
            print(f"No feasible solution found (status={status}).")
            return None, -1

        schedule = [[-1 for _ in range(self.instance.time_slots)] 
                    for _ in range(self.instance.n_operators)]

        for o in range(self.instance.n_operators):
            for t in range(self.instance.time_slots):
                for m in range(self.instance.n_tasks):
                    if pulp.value(self.x[o, t, m]) > 0.5:
                        schedule[o][t] = m

        objective_value = pulp.value(self.model.objective)

        valid, errors = self.instance.validate_schedule(schedule)
        if not valid:
            print(f"WARNING: MILP found a schedule but it's invalid: {errors}")
            return None, -1

        actual_production = self.instance.total_produced(schedule)

        if self.verbose:
            print(f"MILP objective value (instant-flow model) = {objective_value:.2f}")
            print(f"script.py flow-based production          = {actual_production:.2f}")

        return schedule, actual_production

def create_empty_schedule(instance):
    """
    Crée un planning vide avec tous les créneaux initialisés à -1.

    Args:
        instance: L’instance du problème.

    Returns:
        List[List[int]]: Planning vide valide (score nul).
    """

    return [[-1 for _ in range(instance.time_slots)] for _ in range(instance.n_operators)]

def visualize_schedule(instance, schedule, title="Operator Schedule"):
    """
    Affiche visuellement un planning d’opérateurs avec une carte de couleurs.

    Args:
        instance: L’instance du problème.
        schedule (List[List[int]]): Planning à représenter graphiquement.
        title (str): Titre du graphique.
    """
    if not schedule:
        print("Empty schedule")
        return

    cmap = plt.cm.get_cmap('tab10', instance.n_tasks + 1)  
    fig, ax = plt.subplots(figsize=(20, instance.n_operators * 2))

    for o in range(instance.n_operators):
        for t in range(instance.time_slots):
            task = schedule[o][t]
            if task != -1:
                color = cmap(task)
                ax.fill_between([t, t+1], [o, o], [o+1, o+1], color=color, edgecolor='black')

    ax.set_xlabel('Time Slots')
    ax.set_ylabel('Operators')
    ax.set_yticks(np.arange(instance.n_operators) + 0.5)
    ax.set_yticklabels([f'Operator {o}' for o in range(instance.n_operators)])

    legend_elems = [plt.Rectangle((0, 0), 1, 1, color=cmap(i)) for i in range(instance.n_tasks)]
    legend_labels = [f'Task {i}' for i in range(instance.n_tasks)]
    ax.legend(legend_elems, legend_labels, loc='upper right')

    plt.title(title)
    plt.tight_layout()
    plt.show()

def solve_instance(instance_file, time_limit=600, verbose=True):
    """
    Résout une instance donnée du problème via la méthode exacte MILP.

    Args:
        instance_file (str): Chemin du fichier JSON de l’instance.
        time_limit (int): Temps limite de résolution en secondes.
        verbose (bool): Affiche les détails de l'exécution si True.

    Returns:
        tuple: Un planning valide et son score de production (schedule, score).
    """
    if verbose:
        print(f"Loading instance from {instance_file}...")
    instance = Instance(instance_file)

    if verbose:
        print(f"Instance loaded with {instance.n_tasks} tasks, {instance.n_operators} operators, "
              f"{instance.time_slots} time slots")

    problem_size = instance.n_operators * instance.n_tasks * instance.time_slots
    if problem_size > 10000 and verbose:
        print(f"WARNING: Problem size {problem_size} is quite large. Expect extended solve times.")

    solver = ExactMILPSolver(instance, time_limit=time_limit, verbose=verbose)
    solver.build_model()
    schedule, score = solver.solve()

    if schedule is None:
        if verbose:
            print("No valid schedule found. Returning empty schedule and score=0.")
        schedule = create_empty_schedule(instance)
        score = 0

    if verbose:
        print(f"Final production score (script.py) = {score}")

    return schedule, score

def main():
    """
    Fonction principale permettant de résoudre un ensemble d’instances prédéfinies.

    Cette fonction charge chaque instance, exécute la résolution, sauvegarde
    les résultats individuels et agrégés, puis tente une visualisation.
    """

    id_list = [
        '4_2_24_0', '4_2_24_1', '4_2_48_0', '4_2_48_1'
    ]
    time_limit = 600
    results = {}

    for instance_id in id_list:
        instance_file = instance_id + ".json"
        print(f"--- Solving instance {instance_id} ---")
        schedule, score = solve_instance(instance_file, time_limit, verbose=True)
        results[instance_id] = {'schedule': schedule, 'score': score}

        output_file = instance_file.replace('.json', '_solution.json')
        with open(output_file, 'w') as f:
            json.dump({'schedule': schedule, 'score': score}, f, indent=2)
        print(f"Solution for {instance_id} saved to {output_file}\n")

    with open("aggregated_solutions.json", "w") as f:
        json.dump(results, f, indent=2)
    print("All solutions saved to aggregated_solutions.json")

    try:
        visualize_schedule(Instance(id_list[0] + ".json"), results[id_list[0]]['schedule'],
                           title=f"{id_list[0]} Schedule - Score: {results[id_list[0]]['score']:.2f}")
    except Exception as e:
        print(f"Visualization failed: {e}")

if __name__ == "__main__":
    main()