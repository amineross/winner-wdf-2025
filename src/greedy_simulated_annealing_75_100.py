# POUR LES INSTANCES DE 75_* à 100_*
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
    """
    solveur avancé pour le problème de planification d'opérateurs, combinant recuit simulé
    et opérateurs de voisinage multiples avec ajustement dynamique.

    ce solveur est conçu pour les grandes instances où les méthodes exactes deviennent coûteuses.

    auteurs: Amine ROSTANE, Gaël GARNIER, Augustin LOGEAIS.
    """


    def __init__(self, instance: Instance, max_iter=10000, start_temp=5.0, min_temp=0.01,
             cooling_rate=0.99, max_time=None, reheat_threshold=1000, reheat_factor=5.0,
             verbose=True):
        """
        initialise le solveur avec les paramètres du recuit simulé.

        args:
            instance (Instance): instance du problème à résoudre.
            max_iter (int): nombre maximal d’itérations de recuit simulé.
            start_temp (float): température initiale.
            min_temp (float): température minimale (seuil d’arrêt).
            cooling_rate (float): taux de décroissance de la température.
            max_time (float): temps maximal d'exécution en secondes.
            reheat_threshold (int): nombre d’itérations sans amélioration avant réchauffement.
            reheat_factor (float): facteur de réchauffement de la température.
            verbose (bool): active les affichages intermédiaires si vrai.
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
        """
        précalcule les données utiles sur les tâches (niveau de dépendance, opérateurs optimaux, etc.).
        """

        self.critical_tasks = self._identify_critical_tasks()
        self.best_operators_for_task = self._rank_operators_by_productivity()
        self.dependency_levels = self._compute_dependency_levels()

    def _identify_critical_tasks(self):
        """
        identifie les tâches critiques dans le graphe de dépendance (goulots, terminaisons).

        returns:
            set: ensemble des identifiants de tâches critiques.
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
        classe les opérateurs par productivité décroissante pour chaque tâche.

        returns:
            dict: pour chaque tâche, liste ordonnée des opérateurs compatibles.
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
        calcule le niveau topologique de chaque tâche dans le graphe de dépendance.

        returns:
            dict: niveau de dépendance de chaque tâche.
        """

        levels = {}
        graph = self.instance.task_graph

        for node in nx.topological_sort(graph):
            if graph.in_degree(node) == 0:
                levels[node] = 0
            else:
                levels[node] = 1 + max(levels[pred] for pred in graph.predecessors(node))

        return levels

    def simulated_annealing(self, init_schedule):
        """
        applique un recuit simulé avec plusieurs opérateurs de voisinage
        et ajustement dynamique des poids.

        args:
            init_schedule (list[list[int]]): planning initial à améliorer.

        returns:
            tuple: (planning amélioré, score de production associé).
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
        """
        sélectionne un opérateur de voisinage au hasard selon une distribution pondérée.

        args:
            weights (dict): dictionnaire des poids pour chaque opérateur.

        returns:
            str: nom de l’opérateur sélectionné.
        """

        operators = list(weights.keys())
        weights_list = [weights[op] for op in operators]
        return random.choices(operators, weights=weights_list, k=1)[0]

    def _adjust_operator_weights(self, weights, stats):
        """
        ajuste dynamiquement les poids des opérateurs en fonction de leur efficacité.

        args:
            weights (dict): poids actuels.
            stats (dict): statistiques d’utilisation et de performance.
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
        crée un hash léger pour représenter un planning (utilisé pour la liste tabou).

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
        génère un voisin du planning courant selon l'opérateur de voisinage choisi.

        args:
            schedule (list[list[int]]): planning courant.
            operator_type (str): nom de l'opérateur à utiliser.

        returns:
            list[list[int]]: nouveau planning voisin.
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
        modifie un bloc aléatoire du planning en réassignant une tâche ou une période d’inactivité.

        args:
            schedule (list[list[int]]): planning courant.

        returns:
            list[list[int]]: planning modifié.
        """

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
                    chosen_tasks = random.choices(
                        tasks_weighted,
                        weights=weights[:len(tasks_weighted)],
                        k=1
                    )[0]
                else:
                    chosen_tasks = random.choice(tasks_weighted)

                for t in range(start_slot, start_slot + block_len):
                    neighbor[o][t] = chosen_tasks
            else:

                for t in range(start_slot, start_slot + block_len):
                    neighbor[o][t] = -1

        return neighbor

    def _neighbor_swap_blocks(self, schedule):
        """
        échange deux blocs de tâches entre opérateurs si les contraintes le permettent.

        args:
            schedule (list[list[int]]): planning courant.

        returns:
            list[list[int]]: planning modifié.
        """

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
        """
        identifie les blocs continus d'une même tâche dans le planning d’un opérateur.

        args:
            operator_schedule (list[int]): planning d’un opérateur.

        returns:
            list[list[int]]: liste de blocs (listes d’indices).
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
        décale dans le temps une chaîne de tâches dépendantes.

        args:
            schedule (list[list[int]]): planning courant.

        returns:
            list[list[int]]: planning modifié.
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
        extrait des chaînes de tâches dans le graphe de dépendance (chemins simples).

        returns:
            list[list[int]]: liste de chemins (chaînes de tâches).
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
        affecte davantage de ressources aux tâches identifiées comme critiques.

        args:
            schedule (list[list[int]]): planning courant.

        returns:
            list[list[int]]: planning modifié.
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

    def _safe_availability_range(self, operator_idx, schedule):
        """
        retourne la plage de disponibilité d’un opérateur, bornée aux dimensions du planning.

        args:
            operator_idx (int): indice de l’opérateur.
            schedule (list[list[int]]): planning en cours.

        returns:
            tuple: (début, fin) de la plage valide.
        """

        raw_start = self.instance.operators_availability[operator_idx]['start']
        raw_end = self.instance.operators_availability[operator_idx]['end']

        safe_start = max(0, raw_start)
        safe_end = min(len(schedule[operator_idx]) - 1, raw_end)

        return safe_start, safe_end

    def solve(self):
        """
        méthode principale du solveur : génère une solution initiale puis l'améliore.

        returns:
            tuple: (meilleur planning trouvé, score de production associé).
        """


        if self.verbose:
            print("\n[Solver] Building initial greedy schedule...")

        greedy_schedule = generer_planning_productif(self.instance)

        greedy_score = self.instance.total_produced(greedy_schedule)

        print(f"Initial greedy schedule score = {greedy_score:.2f}")

        best_schedule, best_score = self.simulated_annealing(greedy_schedule)

        return best_schedule, best_score

def trouver_chemin_critique(instance): # Passage a des noms de fonction en français désolé on est plusieurs a travailler dessus et pas tous à l'aise avec l'anglais mdr
    """
    identifie un chemin critique dans le graphe de tâches, représentant un flux potentiel maximal.

    args:
        instance (Instance): instance du problème.

    returns:
        list[int]: liste ordonnée d’identifiants de tâches constituant le chemin critique.
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
    affecte les opérateurs aux tâches du chemin critique dans un ordre temporel cohérent.

    cette stratégie cherche à maximiser la continuité du flux dès le début du planning.

    args:
        instance (Instance): instance du problème.
        planning (list[list[int]]): planning partiellement rempli.
        chemin_critique (list[int]): liste des tâches critiques à prioriser.

    returns:
        list[list[int]]: planning mis à jour avec les affectations séquentielles.
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
    renforce les premières tâches du chemin critique en ajoutant des opérateurs supplémentaires.

    args:
        instance (Instance): instance du problème.
        planning (list[list[int]]): planning partiellement rempli.
        chemin_critique (list[int]): liste des tâches critiques.

    returns:
        list[list[int]]: planning modifié avec renforcement des débuts de chaîne.
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
    vérifie et garantit que chaque opérateur respecte le temps de travail minimal requis.

    args:
        instance (Instance): instance du problème.
        planning (list[list[int]]): planning partiellement rempli.

    returns:
        list[list[int]]: planning modifié si nécessaire pour respecter les contraintes.
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
    génère un planning alternatif en mettant l’accent sur un flux continu dès le début.

    utile en cas d’échec ou de production nulle avec la méthode classique.

    args:
        instance (Instance): instance du problème.

    returns:
        list[list[int]]: planning généré via une approche différente.
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
    exécute le solveur sur une instance spécifique.

    cette fonction charge les données de l’instance, configure le solveur et lance la résolution.

    args:
        instance_file (str): chemin du fichier JSON de l'instance.
        max_iter (int): nombre maximal d’itérations pour le recuit simulé.
        max_time (float): temps d'exécution maximal en secondes.
        verbose (bool): active les affichages détaillés si vrai.

    returns:
        tuple: (planning final, score de production).
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

    json_files = glob.glob(f"75*.json")

    for json_file in json_files:
        main(json_file, max_iter=10000, max_time=300, verbose=True)