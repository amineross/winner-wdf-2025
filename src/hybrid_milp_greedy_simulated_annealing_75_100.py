# POUR LES INSTANCES DE 75_* à 100_*
import random
import copy
import math
import time
import networkx as nx
import numpy as np
import glob
import pulp
from collections import defaultdict
from script import Instance


class HybridSchedulingSolver:
    """
    solveur hybride combinant métaheuristiques et optimisation MILP pour la planification d'opérateurs.

    ce solveur alterne entre des modifications locales par recuit simulé et des résolutions exactes par MILP
    sur des fenêtres temporelles, afin d'optimiser la production dans des contraintes industrielles complexes.

    auteurs:
        amine rostane, gaël garnier, augustin logeais
    """


    def __init__(self,
                 instance: Instance,
                 max_iter=10000,
                 start_temp=5.0,
                 min_temp=0.01,
                 cooling_rate=0.99,
                 max_time=None,
                 reheat_threshold=1000,
                 reheat_factor=5.0,
                 milp_time_limit=60,  
                 milp_frequency=2000, 
                 milp_window_size=24, 
                 verbose=True):
        """
        initialise le solveur hybride avec les paramètres du problème et des stratégies de recherche.

        args:
            instance (Instance): instance du problème contenant le graphe des tâches, les opérateurs et les contraintes.
            max_iter (int): nombre maximal d’itérations pour le recuit simulé.
            start_temp (float): température initiale pour le recuit simulé.
            min_temp (float): température minimale avant arrêt du recuit.
            cooling_rate (float): taux de refroidissement de la température.
            max_time (float): temps maximal d'exécution en secondes.
            reheat_threshold (int): nombre d’itérations sans amélioration avant relance thermique.
            reheat_factor (float): facteur de réchauffement de la température.
            milp_time_limit (int): durée maximale (en secondes) pour la résolution MILP d’une fenêtre.
            milp_frequency (int): fréquence (en nombre d’itérations) entre deux appels MILP.
            milp_window_size (int): taille de la fenêtre temporelle utilisée pour MILP.
            verbose (bool): affiche les détails d'exécution si vrai.
        """
        self.instance = instance

        self.max_iter = max_iter
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.max_time = max_time
        self.reheat_threshold = reheat_threshold
        self.reheat_factor = reheat_factor

        self.milp_time_limit = milp_time_limit
        self.milp_frequency = milp_frequency
        self.milp_window_size = min(milp_window_size, instance.time_slots)

        self.verbose = verbose

        self._precompute_task_data()

    def _precompute_task_data(self):
        """
        pré-calcule les structures de données utiles pour les heuristiques :
        - identification des tâches critiques
        - niveaux de dépendance topologiques
        - classement des opérateurs par productivité
        """

        self.critical_tasks = self._identify_critical_tasks()
        self.best_operators_for_task = self._rank_operators_by_productivity()
        self.dependency_levels = self._compute_dependency_levels()

    def _identify_critical_tasks(self):
        """
        identifie les tâches critiques à fort impact sur le flux,
        en analysant les goulots d'étranglement dans le graphe de dépendances.

        returns:
            set[int]: ensemble des identifiants de tâches critiques.
        """

        critical = set()

        for node in self.instance.task_graph.nodes():
            in_degree = self.instance.task_graph.in_degree(node)
            out_degree = self.instance.task_graph.out_degree(node)

            if in_degree > 1 or out_degree > 1:
                critical.add(node)

            if out_degree == 0:
                critical.add(node)

        return critical

    def _rank_operators_by_productivity(self):
        """
        trie les opérateurs compatibles pour chaque tâche selon leur coefficient de productivité.

        returns:
            dict[int, list[int]]: dictionnaire tâche → opérateurs triés par performance décroissante.
        """

        rankings = {}

        for task in range(self.instance.n_tasks):

            compatible_ops = [
                o for o in range(self.instance.n_operators) 
                if (task, o) not in self.instance.incompatibilities
            ]

            ranked_ops = sorted(
                compatible_ops, 
                key=lambda o: self.instance.operators_prod_coef[o][task],
                reverse=True
            )

            rankings[task] = ranked_ops

        return rankings

    def _compute_dependency_levels(self):
        """
        calcule les niveaux topologiques de chaque tâche dans le DAG des dépendances.

        returns:
            dict[int, int]: dictionnaire tâche → niveau (profondeur dans le graphe).
        """

        levels = {}
        graph = self.instance.task_graph

        for node in nx.topological_sort(graph):
            if graph.in_degree(node) == 0:  
                levels[node] = 0
            else:
                levels[node] = 1 + max(levels[pred] for pred in graph.predecessors(node))

        return levels

    def optimize_time_window_milp(self, current_schedule, time_window_start, time_window_size):
        """
        optimise une fenêtre temporelle du planning via MILP en maximisant la production finale.

        args:
            current_schedule (list[list[int]]): planning actuel à optimiser localement.
            time_window_start (int): slot de départ de la fenêtre temporelle.
            time_window_size (int): nombre de slots dans la fenêtre.

        returns:
            list[list[int]]: nouveau planning avec la fenêtre optimisée.
        """

        if self.verbose:
            print(f"Running MILP on time window [{time_window_start}, {time_window_start + time_window_size - 1}]")

        new_schedule = copy.deepcopy(current_schedule)

        time_window_end = min(time_window_start + time_window_size - 1, self.instance.time_slots - 1)

        model = pulp.LpProblem("Time_Window_Optimization", pulp.LpMaximize)

        x = {}
        for o in range(self.instance.n_operators):
            for t in range(time_window_start, time_window_end + 1):
                for m in range(self.instance.n_tasks):
                    x[o,t,m] = pulp.LpVariable(f"x_{o}_{t}_{m}", cat=pulp.LpBinary)

        y = {}
        for o in range(self.instance.n_operators):
            for t in range(time_window_start, time_window_end + 1):
                y[o,t] = pulp.LpVariable(f"y_{o}_{t}", cat=pulp.LpBinary)

        z = {}
        for o in range(self.instance.n_operators):
            for t in range(time_window_start, time_window_end + 1):
                for m in range(self.instance.n_tasks):
                    z[o,t,m] = pulp.LpVariable(f"z_{o}_{t}_{m}", cat=pulp.LpBinary)

        task_production = {}
        for t in range(time_window_start, time_window_end + 1):
            for m in range(self.instance.n_tasks):
                task_production[m,t] = pulp.LpVariable(f"prod_{m}_{t}", lowBound=0)

        task_capacity = {}
        for t in range(time_window_start, time_window_end + 1):
            for m in range(self.instance.n_tasks):
                task_capacity[m,t] = pulp.LpVariable(f"cap_{m}_{t}", lowBound=0)

        leftover = {}
        for m in range(self.instance.n_tasks):

            leftover[m, time_window_start-1] = pulp.LpVariable(f"leftover_{m}_{time_window_start-1}", lowBound=0)
            for t in range(time_window_start, time_window_end + 1):
                leftover[m,t] = pulp.LpVariable(f"leftover_{m}_{t}", lowBound=0)

        last_task = self.instance.n_tasks - 1
        model += pulp.lpSum(task_production[last_task, t] 
                           for t in range(time_window_start, time_window_end + 1))

        for o in range(self.instance.n_operators):
            for t in range(time_window_start, time_window_end + 1):
                model += pulp.lpSum(x[o,t,m] for m in range(self.instance.n_tasks)) + y[o,t] == 1

        for o in range(self.instance.n_operators):
            start_avail = self.instance.operators_availability[o]['start']
            end_avail = self.instance.operators_availability[o]['end']

            for t in range(time_window_start, time_window_end + 1):
                if t < start_avail or t > end_avail:
                    model += y[o,t] == 1  

        for (task, operator) in self.instance.incompatibilities:
            for t in range(time_window_start, time_window_end + 1):
                model += x[operator,t,task] == 0

        for m in range(self.instance.n_tasks):
            base_cap = self.instance.task_capacity[m]
            for t in range(time_window_start, time_window_end + 1):
                model += task_capacity[m,t] == pulp.lpSum(
                    x[o,t,m] * self.instance.operators_prod_coef[o][m]
                    for o in range(self.instance.n_operators)
                ) * base_cap

        for o in range(self.instance.n_operators):
            for m in range(self.instance.n_tasks):
                for t in range(time_window_start, time_window_end + 1):
                    if t == 0 or t == time_window_start:

                        if t > 0:
                            prev_task = current_schedule[o][t-1]
                            if prev_task == m:
                                model += z[o,t,m] == 0  
                            else:
                                model += z[o,t,m] == x[o,t,m]  
                        else:
                            model += z[o,t,m] == x[o,t,m]  
                    else:

                        model += z[o,t,m] >= x[o,t,m] - x[o,t-1,m]
                        model += z[o,t,m] <= x[o,t,m]
                        model += z[o,t,m] <= 1 - x[o,t-1,m]

        min_slots = self.instance.min_session_slots
        for o in range(self.instance.n_operators):
            for m in range(self.instance.n_tasks):

                for t in range(time_window_start, time_window_end + 1):

                    for i in range(1, min_slots):
                        if t + i <= time_window_end:
                            model += x[o,t+i,m] >= z[o,t,m]
                        else:

                            model += z[o,t,m] == 0
                            break

        max_slots = self.instance.max_session_slots
        for o in range(self.instance.n_operators):
            for m in range(self.instance.n_tasks):

                for t in range(time_window_start, time_window_end - max_slots + 2):
                    slide_end = min(t + max_slots, time_window_end + 1)
                    model += pulp.lpSum(x[o,t+i,m] for i in range(slide_end - t)) <= max_slots

        for o in range(self.instance.n_operators):
            for m in range(self.instance.n_tasks):

                existing_count = sum(1 for t in range(self.instance.time_slots) 
                                   if t < time_window_start or t > time_window_end 
                                   and current_schedule[o][t] == m)

                model += pulp.lpSum(x[o,t,m] for t in range(time_window_start, time_window_end + 1)) <= self.instance.max_task_slots - existing_count

        if time_window_start > 0:

            temp_schedule = []
            for o in range(self.instance.n_operators):
                temp_schedule.append(current_schedule[o][:time_window_start])

            task_flows = self._simulate_partial_flow(temp_schedule)

            for m in range(self.instance.n_tasks):
                if m == 0:

                    model += leftover[m, time_window_start-1] == 1000000
                else:

                    model += leftover[m, time_window_start-1] == task_flows[m]
        else:

            for m in range(self.instance.n_tasks):
                if m == 0:
                    model += leftover[m, time_window_start-1] == 1000000
                else:
                    model += leftover[m, time_window_start-1] == 0

        for t in range(time_window_start, time_window_end + 1):
            for m in range(self.instance.n_tasks):

                model += task_production[m,t] <= task_capacity[m,t]

                if m == 0:

                    model += task_production[m,t] <= leftover[m,t-1]
                else:

                    preds = list(self.instance.task_graph.predecessors(m))
                    model += task_production[m,t] <= leftover[m,t-1] + pulp.lpSum(
                        task_production[pred,t] * self.instance.task_graph[pred][m]['weight']
                        for pred in preds
                    )

                if m == 0:
                    model += leftover[m,t] == leftover[m,t-1] - task_production[m,t]
                else:
                    preds = list(self.instance.task_graph.predecessors(m))
                    model += leftover[m,t] == leftover[m,t-1] + pulp.lpSum(
                        task_production[pred,t] * self.instance.task_graph[pred][m]['weight']
                        for pred in preds
                    ) - task_production[m,t]

        for o in range(self.instance.n_operators):

            if time_window_start > 0:

                t_look_back = max(0, time_window_start - self.instance.max_session_slots)

                for m in range(self.instance.n_tasks):
                    session_start = -1
                    session_length = 0

                    for t in range(t_look_back, time_window_start):
                        if current_schedule[o][t] == m:
                            if session_start == -1:
                                session_start = t
                            session_length += 1
                        else:
                            session_start = -1
                            session_length = 0

                    if session_start != -1:

                        remaining_min = max(0, self.instance.min_session_slots - session_length)

                        remaining_max = self.instance.max_session_slots - session_length

                        if remaining_min > 0:
                            for t in range(time_window_start, min(time_window_start + remaining_min, time_window_end + 1)):
                                model += x[o,t,m] == 1

                        if remaining_max <= 0:
                            model += x[o,time_window_start,m] == 0

                        elif remaining_max < (time_window_end - time_window_start + 1):
                            if time_window_start + remaining_max <= time_window_end:
                                model += pulp.lpSum(x[o,t,m] for t in range(time_window_start, time_window_start + remaining_max + 1)) <= remaining_max

        solver = pulp.PULP_CBC_CMD(timeLimit=self.milp_time_limit, msg=0)
        model.solve(solver)

        status = pulp.LpStatus[model.status]
        if status not in ["Optimal", "Feasible"]:
            if self.verbose:
                print(f"MILP subproblem could not be solved: {status}")
            return current_schedule  

        for o in range(self.instance.n_operators):
            for t in range(time_window_start, time_window_end + 1):
                new_schedule[o][t] = -1  
                for m in range(self.instance.n_tasks):
                    if pulp.value(x[o,t,m]) > 0.5:
                        new_schedule[o][t] = m
                        break

        valid, errors = self.instance.validate_schedule(new_schedule)
        if not valid:
            if self.verbose:
                print(f"MILP generated an invalid schedule: {errors[:3]}")
            return current_schedule

        new_score = self.instance.total_produced(new_schedule)
        old_score = self.instance.total_produced(current_schedule)

        if self.verbose:
            print(f"MILP optimization result: {old_score:.2f} -> {new_score:.2f}")

        if new_score >= old_score:
            return new_schedule
        else:

            return current_schedule

    def _simulate_partial_flow(self, partial_schedule):
        """
        simule le flux de production pour une portion partielle du planning.

        args:
            partial_schedule (list[list[int]]): planning tronqué jusqu'à un instant t.

        returns:
            list[float]: quantités restantes produites pour chaque tâche.
        """


        n_tasks = self.instance.n_tasks
        time_slots = len(partial_schedule[0]) if partial_schedule else 0

        task_units = [0] * n_tasks
        if time_slots == 0:
            task_units[0] = 1000000  
            return task_units

        for t in range(time_slots):
            prod_task_units = [0] * n_tasks
            rema_task_units = task_units.copy()

            for operator in range(len(partial_schedule)):
                task = partial_schedule[operator][t]
                if task != -1:

                    previous_tasks = list(self.instance.task_graph.predecessors(task))

                    production = self.instance.task_capacity[task] * self.instance.operators_prod_coef[operator][task]

                    if task != 0:

                        previous_tasks_units = [rema_task_units[p] * self.instance.task_graph[p][task]['weight'] 
                                               for p in previous_tasks]
                        unit_remainings = sum(previous_tasks_units)
                        production = min(unit_remainings, production)

                        prod_task_units[task] += production

                        for p_task in previous_tasks:
                            substitute = min(rema_task_units[p_task] * self.instance.task_graph[p_task][task]['weight'], 
                                           production)
                            production -= substitute
                            rema_task_units[p_task] -= substitute

                            if production == 0:
                                break
                    else:

                        prod_task_units[task] += production

            for task in range(n_tasks):
                task_units[task] = rema_task_units[task] + prod_task_units[task]

        return task_units

    def hybrid_simulated_annealing(self, init_schedule):
        """
        méthode principale de recuit simulé hybride avec optimisation périodique par MILP.

        alterne entre voisinages heuristiques et résolution exacte sur sous-problèmes pour explorer
        efficacement l’espace des plannings valides.

        args:
            init_schedule (list[list[int]]): planning initial généré par une méthode gloutonne.

        returns:
            tuple: (meilleur planning trouvé, score de production associé).
        """

        current_schedule = copy.deepcopy(init_schedule)
        current_score = self.instance.total_produced(current_schedule)
        best_schedule = copy.deepcopy(current_schedule)
        best_score = current_score

        temp = self.start_temp
        iteration = 0
        iterations_since_improvement = 0
        start_time = time.time()

        tabu_list = {}
        tabu_tenure = 100

        operator_weights = {
            'block_reassign': 0.4,
            'swap_blocks': 0.3,
            'task_chain_shift': 0.2,
            'critical_task_focus': 0.1
        }

        operator_stats = {op: {'attempts': 0, 'accepted': 0, 'improvements': 0} 
                          for op in operator_weights}

        if self.verbose:
            print(f"\n[Hybrid SA] Starting score = {best_score:.2f}, temp={temp:.2f}")

        while iteration < self.max_iter and temp > self.min_temp:
            if self.max_time and (time.time() - start_time) > self.max_time:
                if self.verbose:
                    print(f"Time limit reached after {iteration} iterations")
                break

            if iteration > 0 and iteration % self.milp_frequency == 0:

                if self.instance.time_slots <= self.milp_window_size:

                    window_start = 0
                    window_size = self.instance.time_slots
                else:

                    window_options = [
                        0,  
                        self.instance.time_slots // 4,  
                        self.instance.time_slots // 2,  
                    ]
                    window_start = random.choice(window_options)

                    window_size = min(self.milp_window_size, self.instance.time_slots - window_start)

                milp_start_time = time.time()
                if self.verbose:
                    print(f"\nRunning MILP subproblem at iteration {iteration}...")

                milp_schedule = self.optimize_time_window_milp(
                    current_schedule, window_start, window_size
                )

                milp_time = time.time() - milp_start_time
                milp_score = self.instance.total_produced(milp_schedule)

                if milp_score > current_score:

                    current_schedule = milp_schedule
                    current_score = milp_score

                    if milp_score > best_score:
                        best_schedule = copy.deepcopy(milp_schedule)
                        best_score = milp_score

                    iterations_since_improvement = 0

                    if self.verbose:
                        print(f"MILP found better solution: {milp_score:.2f} (in {milp_time:.2f}s)")
                else:
                    if self.verbose:
                        print(f"MILP subproblem did not improve solution ({milp_time:.2f}s)")

            operator = self._select_operator(operator_weights)
            operator_stats[operator]['attempts'] += 1

            neighbor_schedule = self._generate_neighbor(current_schedule, operator)

            schedule_hash = self._hash_schedule(neighbor_schedule)
            if schedule_hash in tabu_list and \
               iteration - tabu_list[schedule_hash] < tabu_tenure:

                iteration += 1
                continue

            valid, _ = self.instance.validate_schedule(neighbor_schedule, fast=True)
            if valid:
                neighbor_score = self.instance.total_produced(neighbor_schedule)
                delta = neighbor_score - current_score

                accept = False
                if delta > 0:

                    accept = True
                    operator_stats[operator]['improvements'] += 1
                    iterations_since_improvement = 0

                    if neighbor_score > best_score:
                        best_schedule = copy.deepcopy(neighbor_schedule)
                        best_score = neighbor_score
                        if self.verbose:
                            print(f" Iter={iteration}, New best: {best_score:.2f}")
                else:

                    accept_prob = math.exp(delta / temp)
                    if random.random() < accept_prob:
                        accept = True

                if accept:
                    current_schedule = copy.deepcopy(neighbor_schedule)
                    current_score = neighbor_score
                    operator_stats[operator]['accepted'] += 1

                    tabu_list[schedule_hash] = iteration

            temp *= self.cooling_rate

            iterations_since_improvement += 1
            if iterations_since_improvement >= self.reheat_threshold:
                old_temp = temp
                temp = min(self.start_temp, temp * self.reheat_factor)
                iterations_since_improvement = 0
                if self.verbose:
                    print(f" Iter={iteration}, Reheating: {old_temp:.4f} -> {temp:.4f}")

                self._adjust_operator_weights(operator_weights, operator_stats)

            iteration += 1

            if self.verbose and iteration % max(1, self.max_iter//20) == 0:
                print(f" Iter={iteration}, Temp={temp:.4f}, "
                      f"Score={current_score:.2f}, Best={best_score:.2f}")

        if self.verbose:
            print("\nOperator statistics:")
            for op, stats in operator_stats.items():
                if stats['attempts'] > 0:
                    accept_rate = stats['accepted'] / stats['attempts'] * 100
                    improve_rate = stats['improvements'] / stats['attempts'] * 100
                    print(f"  {op}: attempts={stats['attempts']}, "
                          f"accepted={accept_rate:.1f}%, improvements={improve_rate:.1f}%")

        return best_schedule, best_score

    def _select_operator(self, weights):
        """
        sélectionne un opérateur de voisinage en fonction des poids actuels.

        args:
            weights (dict[str, float]): dictionnaire opérateur → poids.

        returns:
            str: nom de l'opérateur choisi.
        """

        operators = list(weights.keys())
        weights_list = [weights[op] for op in operators]
        return random.choices(operators, weights=weights_list, k=1)[0]

    def _adjust_operator_weights(self, weights, stats):
        """
        ajuste dynamiquement les poids des opérateurs selon leurs performances observées.

        args:
            weights (dict[str, float]): poids actuels des opérateurs.
            stats (dict[str, dict]): statistiques des opérateurs (améliorations, acceptations...).
        """

        total_improvements = sum(s['improvements'] for s in stats.values())

        if total_improvements == 0:

            for op in weights:
                weights[op] = 1.0 / len(weights)
            return

        min_weight = 0.05
        remaining_weight = 1.0 - min_weight * len(weights)

        for op in weights:
            improvement_rate = stats[op]['improvements'] / total_improvements
            weights[op] = min_weight + improvement_rate * remaining_weight

        total = sum(weights.values())
        for op in weights:
            weights[op] /= total

    def _hash_schedule(self, schedule):
        """
        génère un hash unique pour représenter un planning, utilisé dans la liste taboue.

        args:
            schedule (list[list[int]]): planning à encoder.

        returns:
            int: valeur de hachage.
        """


        hash_val = 0
        for i, operator_schedule in enumerate(schedule):
            for j, task in enumerate(operator_schedule):
                if task != -1:  
                    hash_val = (hash_val * 31 + hash((i, j, task))) % (10**9 + 7)
        return hash_val

    def _generate_neighbor(self, schedule, operator_type):
        """
        génère un voisin du planning actuel selon un opérateur spécifié.

        args:
            schedule (list[list[int]]): planning courant.
            operator_type (str): nom de l’opérateur de voisinage.

        returns:
            list[list[int]]: planning modifié (voisin).
        """

        if operator_type == 'block_reassign':
            return self._neighbor_block_reassign(schedule)
        elif operator_type == 'swap_blocks':
            return self._neighbor_swap_blocks(schedule)
        elif operator_type == 'task_chain_shift':
            return self._neighbor_task_chain_shift(schedule)
        elif operator_type == 'critical_task_focus':
            return self._neighbor_critical_task_focus(schedule)
        else:

            return self._neighbor_block_reassign(schedule)

    def _neighbor_block_reassign(self, schedule):
        """
        génère un voisin du planning en réaffectant aléatoirement un bloc d’un opérateur.

        la tâche assignée dans le bloc est choisie en fonction :
        - de la productivité de l’opérateur
        - de la criticité de la tâche
        - de sa position dans le graphe de dépendances

        args:
            schedule (list[list[int]]): planning courant.

        returns:
            list[list[int]]: planning modifié avec un bloc potentiellement réassigné.
        """

        neighbor = copy.deepcopy(schedule)
        O = self.instance.n_operators
        T = self.instance.time_slots
        M = self.instance.n_tasks

        o = random.randint(0, O - 1)

        raw_start = self.instance.operators_availability[o]['start']
        raw_end = self.instance.operators_availability[o]['end']
        start_avail = max(0, min(raw_start, T - 1))
        end_avail = min(T - 1, raw_end)

        if end_avail < start_avail:
            return neighbor

        min_ses = self.instance.min_session_slots
        max_ses = self.instance.max_session_slots

        available_length = end_avail - start_avail + 1
        if available_length < min_ses:
            return neighbor

        block_len = random.randint(min_ses, min(max_ses, available_length))

        start_slot = random.randint(start_avail, end_avail - block_len + 1)

        if random.random() < 0.3:  
            for t in range(start_slot, start_slot + block_len):
                neighbor[o][t] = -1
        else:

            feasible_tasks = [
                m for m in range(M)
                if (m, o) not in self.instance.incompatibilities
            ]

            if feasible_tasks:

                tasks_weighted = sorted(
                    feasible_tasks,
                    key=lambda m: (

                        -self.instance.operators_prod_coef[o][m],

                        self.dependency_levels.get(m, 0),

                        m not in self.critical_tasks
                    )
                )

                if len(tasks_weighted) > 3:
                    weights = [1.0, 0.5, 0.3] + [0.1] * (len(tasks_weighted) - 3)
                    chosen_task = random.choices(
                        tasks_weighted, 
                        weights=weights[:len(tasks_weighted)],
                        k=1
                    )[0]
                else:
                    chosen_task = random.choice(tasks_weighted)

                for t in range(start_slot, start_slot + block_len):
                    neighbor[o][t] = chosen_task
            else:

                for t in range(start_slot, start_slot + block_len):
                    neighbor[o][t] = -1

        return neighbor

    def _neighbor_swap_blocks(self, schedule):
        """
        échange deux blocs de tâches entre deux opérateurs, si compatible.

        cette opération permet de redistribuer les ressources sans casser le planning global.

        args:
            schedule (list[list[int]]): planning courant.

        returns:
            list[list[int]]: planning avec deux blocs échangés si possible.
        """

        neighbor = copy.deepcopy(schedule)
        O = self.instance.n_operators

        if O < 2:
            return neighbor

        o1 = random.randint(0, O - 1)
        o2 = random.randint(0, O - 1)
        while o2 == o1:
            o2 = random.randint(0, O - 1)

        blocks_o1 = self._find_blocks(schedule[o1])
        if not blocks_o1:
            return neighbor

        blocks_o2 = self._find_blocks(schedule[o2])
        if not blocks_o2:
            return neighbor

        block1 = random.choice(blocks_o1)
        block2 = random.choice(blocks_o2)

        task1 = schedule[o1][block1[0]]
        task2 = schedule[o2][block2[0]]

        if (task1, o2) in self.instance.incompatibilities or \
           (task2, o1) in self.instance.incompatibilities:
            return neighbor

        o1_start = max(0, self.instance.operators_availability[o1]['start'])
        o1_end = min(len(schedule[o1]) - 1, self.instance.operators_availability[o1]['end'])
        o2_start = max(0, self.instance.operators_availability[o2]['start'])
        o2_end = min(len(schedule[o2]) - 1, self.instance.operators_availability[o2]['end'])

        for t in block2:
            if t < o1_start or t > o1_end:
                return neighbor

        for t in block1:
            if t < o2_start or t > o2_end:
                return neighbor

        for t in block1:
            neighbor[o2][t] = task1

        for t in block2:
            neighbor[o1][t] = task2

        return neighbor

    def _find_blocks(self, operator_schedule):
        """
        identifie les blocs consécutifs de tâches identiques dans l’emploi du temps d’un opérateur.

        args:
            operator_schedule (list[int]): liste des tâches assignées à un opérateur au fil du temps.

        returns:
            list[list[int]]: liste de blocs, chaque bloc étant une liste d’indices de slots consécutifs.
        """

        blocks = []
        current_task = -2  
        current_block = []

        for t, task in enumerate(operator_schedule):
            if task == current_task and task != -1:
                current_block.append(t)
            else:
                if current_block and current_task != -1:
                    blocks.append(current_block)
                current_block = [t] if task != -1 else []
                current_task = task

        if current_block and current_task != -1:
            blocks.append(current_block)

        return blocks

    def _neighbor_task_chain_shift(self, schedule):
        """
        déplace une chaîne de tâches dépendantes vers la gauche ou la droite dans le temps.

        cette mutation respecte les dépendances et explore d’autres synchronisations.

        args:
            schedule (list[list[int]]): planning courant.

        returns:
            list[list[int]]: planning avec une sous-chaîne de tâches déplacée.
        """

        neighbor = copy.deepcopy(schedule)

        chains = self._find_task_chains()
        if not chains:
            return self._neighbor_block_reassign(schedule)  

        chain = random.choice(chains)
        if len(chain) < 2:
            return self._neighbor_block_reassign(schedule)  

        start_idx = random.randint(0, len(chain) - 2)
        sub_chain = chain[start_idx:start_idx + min(3, len(chain) - start_idx)]

        shift_earlier = random.choice([True, False])
        shift_amount = random.randint(1, 5)  

        for task in sub_chain:

            for o in range(self.instance.n_operators):
                if (task, o) in self.instance.incompatibilities:
                    continue

                blocks = []
                current_block = []

                for t, assigned_task in enumerate(schedule[o]):
                    if assigned_task == task:
                        current_block.append(t)
                    else:
                        if current_block:
                            blocks.append(current_block)
                            current_block = []

                if current_block:
                    blocks.append(current_block)

                for block in blocks:

                    o_start = max(0, self.instance.operators_availability[o]['start'])
                    o_end = min(len(schedule[o]) - 1, self.instance.operators_availability[o]['end'])

                    if shift_earlier:
                        new_start = max(o_start, block[0] - shift_amount)
                    else:
                        max_valid_start = min(o_end - len(block) + 1, len(schedule[o]) - len(block))
                        new_start = min(max_valid_start, block[0] + shift_amount)

                    if new_start < o_start or new_start + len(block) - 1 > o_end:
                        continue

                    for t in block:
                        neighbor[o][t] = -1

                    for i, _ in enumerate(block):
                        if new_start + i < len(schedule[o]):
                            neighbor[o][new_start + i] = task

        return neighbor

    def _find_task_chains(self):
        """
        extrait les chaînes de dépendances (du graphe) reliant les sources aux puits.

        returns:
            list[list[int]]: ensemble de chemins de tâches ordonnés dans le graphe de dépendances.
        """

        chains = []

        sources = [n for n in self.instance.task_graph.nodes() 
                   if self.instance.task_graph.in_degree(n) == 0]
        sinks = [n for n in self.instance.task_graph.nodes() 
                 if self.instance.task_graph.out_degree(n) == 0]

        for source in sources:
            for sink in sinks:
                try:
                    for path in nx.all_simple_paths(self.instance.task_graph, source, sink):
                        chains.append(path)
                except nx.NetworkXNoPath:
                    continue

        return chains

    def _neighbor_critical_task_focus(self, schedule):
        """
        améliore le planning en ajoutant ou renforçant une tâche critique dans des créneaux disponibles.

        args:
            schedule (list[list[int]]): planning courant.

        returns:
            list[list[int]]: planning modifié avec accent sur une tâche critique.
        """

        neighbor = copy.deepcopy(schedule)

        if not self.critical_tasks:
            return self._neighbor_block_reassign(schedule)

        task = random.choice(list(self.critical_tasks))

        best_ops = self.best_operators_for_task.get(task, [])
        if not best_ops:
            return self._neighbor_block_reassign(schedule)

        op_candidates = best_ops[:min(3, len(best_ops))]
        o = random.choice(op_candidates)

        available_slots = []

        o_start = max(0, self.instance.operators_availability[o]['start'])
        o_end = min(len(schedule[o]) - 1, self.instance.operators_availability[o]['end'])

        if o_start > o_end:  
            return self._neighbor_block_reassign(schedule)

        for t in range(o_start, o_end + 1):
            current_task = schedule[o][t]

            if current_task == -1 or current_task not in self.critical_tasks:
                available_slots.append(t)

        if not available_slots:
            return self._neighbor_block_reassign(schedule)

        blocks = []
        current_block = []
        min_ses = self.instance.min_session_slots

        for t in sorted(available_slots):
            if not current_block or t == current_block[-1] + 1:
                current_block.append(t)
            else:
                if len(current_block) >= min_ses:
                    blocks.append(current_block)
                current_block = [t]

        if current_block and len(current_block) >= min_ses:
            blocks.append(current_block)

        if not blocks:
            return self._neighbor_block_reassign(schedule)

        block = random.choice(blocks)

        max_ses = self.instance.max_session_slots
        if len(block) > max_ses:
            start_idx = random.randint(0, len(block) - max_ses)
            block = block[start_idx:start_idx + max_ses]

        for t in block:
            neighbor[o][t] = task

        return neighbor

    def solve(self):
        """
        méthode principale du solveur hybride.

        elle combine une génération gloutonne initiale avec un recuit simulé enrichi par des
        optimisations MILP ponctuelles sur des fenêtres temporelles.

        returns:
            tuple: (meilleur planning trouvé, score total de production associé).
        """


        if self.verbose:
            print("\n[Solver] Building initial greedy schedule...")

        greedy_schedule = generer_planning_productif(self.instance)
        greedy_score = self.instance.total_produced(greedy_schedule)
        print(f"Initial greedy schedule score = {greedy_score:.2f}")

        best_schedule, best_score = self.hybrid_simulated_annealing(greedy_schedule)

        return best_schedule, best_score

def trouver_chemin_critique(instance):
    """
    Identifie le chemin avec le meilleur potentiel de production
    """
    G = instance.task_graph
    source = 0  
    target = instance.n_tasks - 1  

    try:

        path = nx.shortest_path(G, source=source, target=target, 
                              weight=lambda u, v, d: -d['weight'] * instance.task_capacity[v])
        return path
    except:

        paths = list(nx.all_simple_paths(G, source, target))
        if paths:
            return paths[0]
        return []

def assigner_operateurs_sequentiellement(instance, planning, chemin_critique):
    """
    Assigne des opérateurs aux tâches du chemin critique de manière séquentielle
    en tenant compte de l'ordre temporel des tâches
    """

    time_span = instance.time_slots - 10  
    task_spacing = time_span / len(chemin_critique)

    for idx, tache in enumerate(chemin_critique):

        slot_optimal = int(5 + idx * task_spacing)  

        operateurs_compatibles = [op for op in range(instance.n_operators) 
                                 if tache in instance.ope_task[op]]

        if not operateurs_compatibles:
            continue

        assigned = False
        for window in range(5):  
            for direction in [0, 1, -1]:  
                slot = slot_optimal + direction * window

                if slot < 0 or slot >= instance.time_slots - instance.min_session_slots:
                    continue

                for op in operateurs_compatibles:

                    if slot < instance.operators_availability[op]['start'] or slot + instance.min_session_slots > instance.operators_availability[op]['end']:
                        continue

                    if all(planning[op][slot+i] == -1 for i in range(instance.min_session_slots)):

                        duree = min(instance.min_session_slots * 2, 
                                  instance.max_session_slots,
                                  instance.operators_availability[op]['end'] - slot)

                        for i in range(duree):
                            planning[op][slot+i] = tache

                        assigned = True
                        break

                if assigned:
                    break

            if assigned:
                break

    return planning

def renforcer_taches_critiques(instance, planning, chemin_critique):
    """
    Ajoute des opérateurs supplémentaires aux tâches critiques
    pour assurer un flux suffisant
    """

    taches_prioritaires = chemin_critique[:len(chemin_critique)//3]

    for tache in taches_prioritaires:

        operateurs_compatibles = []
        for op in range(instance.n_operators):
            if tache in instance.ope_task[op] and tache not in planning[op]:
                operateurs_compatibles.append(op)

        if not operateurs_compatibles:
            continue

        for op in operateurs_compatibles:
            start = instance.operators_availability[op]['start']
            end = instance.operators_availability[op]['end']

            for slot in range(start, end - instance.min_session_slots + 1):
                if all(planning[op][slot+i] == -1 for i in range(instance.min_session_slots)):

                    duree = min(instance.min_session_slots + 5, instance.max_session_slots)
                    for i in range(duree):
                        if slot+i < end:
                            planning[op][slot+i] = tache
                    break

    return planning

def assurer_temps_minimal(instance, planning):
    """
    S'assure que chaque opérateur respecte son temps de travail minimal
    """
    for op in range(instance.n_operators):

        temps_travail = sum(1 for slot in planning[op] if slot != -1)

        if temps_travail < instance.min_work_slots:

            start = instance.operators_availability[op]['start']
            end = instance.operators_availability[op]['end']
            taches_compatibles = instance.ope_task[op]

            slots_manquants = instance.min_work_slots - temps_travail
            slot_actuel = start

            while slots_manquants > 0 and slot_actuel <= end - instance.min_session_slots:

                if all(planning[op][slot_actuel+i] == -1 for i in range(instance.min_session_slots)):

                    for tache in taches_compatibles:
                        compteur_tache = planning[op].count(tache)
                        if compteur_tache < instance.max_task_slots:

                            duree = min(instance.min_session_slots, 
                                      slots_manquants,
                                      instance.max_task_slots - compteur_tache)
                            for i in range(duree):
                                planning[op][slot_actuel+i] = tache

                            slots_manquants -= duree
                            slot_actuel += duree
                            break
                    else:
                        slot_actuel += 1
                else:
                    slot_actuel += 1

    return planning

def generer_planning_alternatif(instance):
    """
    Approche alternative qui privilégie le flux continu
    """
    planning = [[-1 for _ in range(instance.time_slots)] for _ in range(instance.n_operators)]

    G = instance.task_graph
    source = 0
    target = instance.n_tasks - 1

    try:
        chemin = nx.shortest_path(G, source, target)  
    except:
        return planning

    premieres_taches = chemin[:3]
    for tache in premieres_taches:
        ops_compatibles = [op for op in range(instance.n_operators) 
                          if tache in instance.ope_task[op]]

        for op in ops_compatibles[:5]:  
            start = instance.operators_availability[op]['start']
            end = min(start + 50, instance.operators_availability[op]['end'])

            for t in range(start, end):
                planning[op][t] = tache

    for i, tache in enumerate(chemin[3:], 3):
        start_time = i * 15  

        ops_compatibles = [op for op in range(instance.n_operators) 
                          if tache in instance.ope_task[op]]

        for op in ops_compatibles[:3]:  
            start = max(start_time, instance.operators_availability[op]['start'])
            end = min(start + 30, instance.operators_availability[op]['end'])

            if all(planning[op][t] == -1 for t in range(start, end)):
                for t in range(start, end):
                    planning[op][t] = tache

    planning = assurer_temps_minimal(instance, planning)

    return planning

def generer_planning_productif(instance):
    """
    Génère un planning qui garantit une production positive
    """

    planning = [[-1 for _ in range(instance.time_slots)] for _ in range(instance.n_operators)]

    chemin = trouver_chemin_critique(instance)
    if not chemin:
        return planning

    planning = assigner_operateurs_sequentiellement(instance, planning, chemin)

    planning = renforcer_taches_critiques(instance, planning, chemin)

    planning = assurer_temps_minimal(instance, planning)

    valid, errors = instance.validate_schedule(planning)

    if valid:
        flux = instance.simulate_task_flow(planning)
        production = flux[-1]
        print(f"Planning valide avec production: {production}")

        if production == 0:
            print("Production nulle, essayons une autre approche...")
            return generer_planning_alternatif(instance)
    else:
        print("Planning non valide, erreurs:", errors)

    return planning

def main(instance_file, max_iter=50000, max_time=600, verbose=True):
    """
    Main function to run the solver on a given instance
    """
    print(f"\n=== Running Hybrid Metaheuristic+MILP on {instance_file} ===")
    instance = Instance(instance_file)

    instance_size = f"{instance.n_tasks} tasks, {instance.n_operators} operators, {instance.time_slots} slots"
    print(f"Instance size: {instance_size}")

    temperature = 10.0 if instance.n_tasks > 20 else 5.0
    cooling_rate = 0.999 if instance.n_tasks > 50 else 0.995
    reheat_threshold = max(1000, instance.n_tasks * 50)

    problem_size = instance.n_operators * instance.n_tasks * instance.time_slots
    milp_time_limit = 30
    milp_window_size = 24
    milp_frequency = 2000

    if problem_size < 5000:
        milp_window_size = min(48, instance.time_slots)
        milp_time_limit = 60
        milp_frequency = 1000

    elif problem_size < 20000:
        milp_window_size = min(36, instance.time_slots)
        milp_time_limit = 45
        milp_frequency = 1500

    else:
        milp_window_size = min(24, instance.time_slots)
        milp_time_limit = 30
        milp_frequency = 2000

    solver = HybridSchedulingSolver(
        instance,
        max_iter=max_iter,
        start_temp=temperature,
        min_temp=0.01,
        cooling_rate=cooling_rate,
        max_time=max_time,
        reheat_threshold=reheat_threshold,
        reheat_factor=5.0,
        milp_time_limit=milp_time_limit,
        milp_frequency=milp_frequency,
        milp_window_size=milp_window_size,
        verbose=verbose
    )

    start_time = time.time()
    schedule, score = solver.solve()
    solve_time = time.time() - start_time

    valid, errors = instance.validate_schedule(schedule)
    if valid:
        final_score = instance.total_produced(schedule)
        print(f"\n✅ Valid schedule found with final production = {final_score:.2f}")
        print(f"Solved in {solve_time:.1f} seconds")
        print(schedule)
    else:
        print("\n❌ The returned schedule is invalid.")
        for e in errors[:5]:  
            print("  -", e)

    return schedule, score

if __name__ == "__main__":

    json_files = glob.glob("75_*.json")

    for json_file in sorted(json_files):
        main(json_file, max_iter=20000, max_time=300, verbose=True)