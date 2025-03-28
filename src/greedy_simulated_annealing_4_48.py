import random
import copy
import math
import time
import networkx as nx
import numpy as np
import glob
from collections import defaultdict
from script import Instance


class AdvancedMetaheuristicSolver:
    """Solveur avancé pour la planification des opérateurs utilisant des métaheuristiques.

    Combine une construction gloutonne améliorée et un recuit simulé renforcé
    avec plusieurs opérateurs de voisinage, une gestion adaptative de la température
    et des mécanismes de réchauffement.

    Auteurs:
        Amine ROSTANE, Gaël GARNIER, Augustin LOGEAIS.
    """

    def __init__(self, instance: Instance, max_iter=10000, start_temp=5.0, min_temp=0.01,
             cooling_rate=0.99, max_time=None, reheat_threshold=1000, reheat_factor=5.0,
             verbose=True):
        """Initialise le solveur métaheuristique avec ses paramètres.

        Args:
            instance (Instance): Données du problème.
            max_iter (int): Nombre maximal d'itérations du recuit simulé.
            start_temp (float): Température initiale.
            min_temp (float): Température minimale (critère d'arrêt).
            cooling_rate (float): Taux de refroidissement à chaque itération.
            max_time (float): Durée maximale d'exécution (en secondes).
            reheat_threshold (int): Seuil d’itérations sans amélioration avant réchauffement.
            reheat_factor (float): Multiplicateur de température lors du réchauffement.
            verbose (bool): Affichage des logs d'exécution.
        """

        self.instance = instance
        self.max_iter = max_iter
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.cooling_rate = cooling_rate
        self.max_time = max_time
        self.reheat_threshold = reheat_threshold
        self.reheat_factor = reheat_factor
        self.verbose = verbose

        self._precompute_task_data()

    def _precompute_task_data(self):
        """Prépare les données utiles pour accélérer les décisions du solveur."""

        self.critical_tasks = self._identify_critical_tasks()
        self.best_operators_for_task = self._rank_operators_by_productivity()
        self.dependency_levels = self._compute_dependency_levels()

    def _identify_critical_tasks(self):
        """Identifie les tâches critiques dans le graphe de dépendances."""

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
        """Classe les opérateurs selon leur productivité pour chaque tâche."""

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
        """Calcule les niveaux topologiques des tâches dans le graphe de dépendance."""

        levels = {}
        graph = self.instance.task_graph

        for node in nx.topological_sort(graph):
            if graph.in_degree(node) == 0:  
                levels[node] = 0
            else:
                levels[node] = 1 + max(levels[pred] for pred in graph.predecessors(node))

        return levels

    def build_greedy_schedule(self):
        """Construit un planning initial valide en respectant l’ordre topologique des tâches.

        Retourne un planning glouton valide si possible, sinon un planning vide.

        Returns:
            tuple: (schedule, valid) où:
                - schedule (List[List[int]]): Planning des opérateurs.
                - valid (bool): Indique si le planning est valide.
        """

        nO = self.instance.n_operators
        nT = self.instance.time_slots
        nM = self.instance.n_tasks

        schedule = [[-1]*nT for _ in range(nO)]

        try:
            topo_tasks = list(nx.topological_sort(self.instance.task_graph))
        except:

            topo_tasks = list(range(nM))

        band_width = max(1, (nT + nM - 1)//nM)  

        op_task_usage = [[0]*nM for _ in range(nO)]

        availability = []
        for o in range(nO):
            start_avail = max(0, self.instance.operators_availability[o]["start"])
            end_avail   = min(nT-1, self.instance.operators_availability[o]["end"])
            availability.append((start_avail, end_avail))

        min_session = self.instance.min_session_slots
        max_session = self.instance.max_session_slots
        max_task_slots = self.instance.max_task_slots

        for i, task in enumerate(topo_tasks):
            band_start = i * band_width
            band_end   = min(nT-1, (i+1)*band_width - 1)
            if band_start > band_end:

                break

            feasible_ops = []
            for o in range(nO):
                if (task, o) not in self.instance.incompatibilities:
                    feasible_ops.append(o)

            for o in feasible_ops:
                ostart, oend = availability[o]

                start_slot = max(ostart, band_start)
                end_slot   = min(oend, band_end)
                if start_slot > end_slot:
                    continue

                length_candidates = [min_session, max_session]

                slot = start_slot
                while slot <= end_slot - min_session + 1:

                    length_ = min_session

                    while length_ <= max_session:
                        if op_task_usage[o][task] + length_ <= max_task_slots and \
                          slot + length_ - 1 <= end_slot:

                            can_assign = True
                            for t_ in range(slot, slot+length_):
                                if schedule[o][t_] != -1:
                                    can_assign = False
                                    break
                            if can_assign:

                                for t_ in range(slot, slot+length_):
                                    schedule[o][t_] = task
                                op_task_usage[o][task] += length_

                                slot += length_ + 1  
                                break
                        length_ += 1
                    else:

                        slot += 1  

        for o in range(nO):
            minw = self.instance.min_work_slots
            current_w = sum(1 for t in range(nT) if schedule[o][t] != -1)
            needed = minw - current_w
            if needed <= 0:
                continue

            for slot in range(nT):
                if needed <= 0:
                    break
                if schedule[o][slot] == -1:

                    for task in range(nM):
                        if (task, o) not in self.instance.incompatibilities:
                            if op_task_usage[o][task] < max_task_slots:

                                schedule[o][slot] = task
                                op_task_usage[o][task] += 1
                                needed -= 1
                                break

        valid, errors = self.instance.validate_schedule(schedule)
        if not valid:
            return ([[-1]*nT for _ in range(nO)], False)
        return (schedule, True)

    def simulated_annealing(self, init_schedule):
        """Améliore un planning initial avec un recuit simulé multi-voisinages.

        Utilise plusieurs opérateurs de voisinage, une gestion adaptative
        de la température, un mécanisme de réchauffement, et une liste taboue.

        Args:
            init_schedule (List[List[int]]): Planning de départ.

        Returns:
            tuple: (best_schedule, best_score) représentant le meilleur planning trouvé
                  et sa production totale.
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
            print(f"\n[Simulated Annealing] Starting score = {best_score:.2f}, temp={temp:.2f}")

        while iteration < self.max_iter and temp > self.min_temp:
            if self.max_time and (time.time() - start_time) > self.max_time:
                if self.verbose:
                    print(f"Time limit reached after {iteration} iterations")
                break

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

            if self.verbose and iteration % max(1, self.max_iter//1000) == 0:
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
        """Sélectionne un opérateur de voisinage selon une distribution pondérée.

        Args:
            weights (dict): Dictionnaire {opérateur: poids}.

        Returns:
            str: Nom de l’opérateur sélectionné.
        """

        operators = list(weights.keys())
        weights_list = [weights[op] for op in operators]
        return random.choices(operators, weights=weights_list, k=1)[0]

    def _adjust_operator_weights(self, weights, stats):
        """Ajuste dynamiquement les poids des opérateurs selon leurs performances."""

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
        """Crée un hash simple d’un planning pour la gestion taboue.

        Args:
            schedule (List[List[int]]): Planning à hasher.

        Returns:
            int: Valeur de hash unique.
        """

        hash_val = 0
        for i, operator_schedule in enumerate(schedule):
            for j, task in enumerate(operator_schedule):
                if task != -1:  
                    hash_val = (hash_val * 31 + hash((i, j, task))) % (10**9 + 7)
        return hash_val

    def _generate_neighbor(self, schedule, operator_type):
        """Génère un planning voisin à l’aide d’un opérateur spécifié.

        Args:
            schedule (List[List[int]]): Planning courant.
            operator_type (str): Type d’opérateur à utiliser.

        Returns:
            List[List[int]]: Planning modifié.
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
        """Réaffecte aléatoirement un bloc de créneaux pour un opérateur."""

        neighbor = copy.deepcopy(schedule)
        O = self.instance.n_operators
        T = self.instance.time_slots
        M = self.instance.n_tasks

        o = random.randint(0, O - 1)

        raw_start = self.instance.operators_availability[o]['start']
        raw_end = self.instance.operators_availability[o]['end']
        start_avail = max(0, min(raw_start, T - 1))
        end_avail = max(0, min(raw_end, T - 1))

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
        """Échange des blocs entiers de tâches entre deux opérateurs."""

        neighbor = copy.deepcopy(schedule)
        O = self.instance.n_operators
        T = self.instance.time_slots

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
        """Trouve les blocs continus de même tâche dans le planning d’un opérateur."""

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
        """Décale des chaînes de tâches dans le planning selon leur dépendance."""

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
        """Recherche toutes les chaînes simples de dépendance entre tâches."""

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
        """Focalise l’optimisation sur les tâches critiques identifiées."""

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

    def _safe_availability_range(self, operator_idx, schedule):
        """Retourne l’intervalle de disponibilité valide d’un opérateur donné."""

        raw_start = self.instance.operators_availability[operator_idx]['start']
        raw_end = self.instance.operators_availability[operator_idx]['end']

        safe_start = max(0, raw_start)
        safe_end = min(len(schedule[operator_idx]) - 1, raw_end)

        return safe_start, safe_end

    def solve(self):
        """Méthode principale combinant construction gloutonne et recuit simulé.

        Returns:
            tuple: (schedule, score) représentant le meilleur planning trouvé
                  et sa production estimée.
        """


        if self.verbose:
            print("\n[Solver] Building initial greedy schedule...")

        greedy_schedule, success = self.build_greedy_schedule()
        if success:
            greedy_score = self.instance.total_produced(greedy_schedule)
            if self.verbose:
                print(f"Initial greedy schedule score = {greedy_score:.2f}")
        else:
            greedy_schedule = [[-1]*self.instance.time_slots for _ in range(self.instance.n_operators)]
            greedy_score = 0
            if self.verbose:
                print("Using empty schedule as initialization (score = 0)")

        best_schedule, best_score = self.simulated_annealing(greedy_schedule)

        return best_schedule, best_score

def main(instance_file, max_iter=50000, max_time=600, verbose=True):
    """Fonction principale d’exécution du solveur sur un fichier JSON donné.

    Args:
        instance_file (str): Chemin vers le fichier de l’instance.
        max_iter (int): Nombre maximal d’itérations de recuit simulé.
        max_time (int): Durée maximale d’exécution (en secondes).
        verbose (bool): Affichage des logs pendant la résolution.

    Returns:
        tuple: (schedule, score) résultat final du solveur.
    """

    print(f"\n=== Running Advanced Metaheuristic on {instance_file} ===")
    instance = Instance(instance_file)

    instance_size = f"{instance.n_tasks} tasks, {instance.n_operators} operators, {instance.time_slots} slots"
    print(f"Instance size: {instance_size}")

    temperature = 10.0 if instance.n_tasks > 20 else 5.0
    cooling_rate = 0.999 if instance.n_tasks > 50 else 0.995
    reheat_threshold = max(1000, instance.n_tasks * 50)

    solver = AdvancedMetaheuristicSolver(
        instance,
        max_iter=max_iter,
        start_temp=temperature,
        min_temp=0.01,
        cooling_rate=cooling_rate,
        max_time=max_time,
        reheat_threshold=reheat_threshold,
        reheat_factor=5.0,
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

    json_files = glob.glob(f"*.json")    

    for json_file in json_files:
        main(json_file, max_iter=10000, max_time=300, verbose=True)