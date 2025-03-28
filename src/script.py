import networkx as nx
from typing import Optional
from typing_extensions import Self
import numpy as np
import pandas as pd
import json
import ast

class Instance:
    def __init__(self, file_path: Optional[str] = None):
        if file_path is not None:
            self._load_json_instance(file_path)
            self._build_task_graph()

    def _load_json_instance(self, file_path: str) -> Self:
        """Charge les données d'une instance JSON."""
        with open(file_path, 'r') as file:
            dict_instance = json.load(file)

        self.load_dict(dict_instance)

    def load_dict(self, dict_instance: dict) -> Self:
        """Charge l'instance à partir d'un dictionnaire"""
        self.n_tasks = int(dict_instance["n_task"])
        self.n_operators = int(dict_instance["n_operators"])
        self.time_slots = int(dict_instance["time_slots"])
        self.min_session_slots = int(dict_instance["min_session_slots"])
        self.max_session_slots = int(dict_instance["max_session_slots"])
        self.min_work_slots = int(dict_instance["min_work_slots"])
        self.max_task_slots = int(dict_instance["max_task_slots"])

        self.operators_availability = [
            {
                "id": op["id"],
                "start": int(op["start"]),
                "end": int(op["end"])
            } for op in dict_instance["operators_availability"]
        ]
        self.task_dependencies = [
            (
                dep["first_node"],
                dep["passed"] / 100.0,
                dep["second_node"]
            ) for dep in dict_instance["task_dependencies"]
        ]

        self.operators_prod_coef = dict_instance["operators_prod_coef"]
        self.task_capacity = dict_instance["task_capacity"]
        self.incompatibilities = [tuple(inc) for inc in dict_instance["incompatibilities"]]

        self.ope_task = []
        for ope in range(self.n_operators):
            ope_incomp = [t for (t, o) in self.incompatibilities if o == ope]
            ope_comp = [t for t in range(self.n_tasks) if t not in ope_incomp]
            self.ope_task.append(ope_comp)

        self._build_task_graph()
        return self

    def _build_task_graph(self):
        """Construit un graphe orienté des dépendances de tâches."""
        self.task_graph = nx.DiGraph()

        for first_node, passed, second_node in self.task_dependencies:
            self.task_graph.add_edge(first_node, second_node, weight=passed)

    def validate_schedule(self, schedule: list[list[int]], fast: bool=False) -> tuple[bool, list[str]]:
        """Vérifie la validité d'un planning et renvoie les différentes erreurs."""
        valid = True
        errors = []
        matrix = np.array(schedule)

        if len(schedule) == 0:
            return valid, errors # planning vide

        if matrix.shape != (self.n_operators, self.time_slots):
            errors.append(f"Le planning doit être de dimensions {(self.n_operators, self.time_slots)}.")
            return False, errors

        for operator in range(self.n_operators):
            operator_schedule = schedule[operator]
            task_counts = {task: 0 for task in range(self.n_tasks)}
            current_task = -1
            current_session_length = 0
            work_slots = 0

            for t in range(self.time_slots):
                task = operator_schedule[t]

                # Vérification des disponibilités
                if task != -1 and (t < self.operators_availability[operator]['start'] or t > self.operators_availability[operator]['end']):
                    errors.append(f"Opérateur {operator} assigne à la tâche {task} hors de ses disponibilités au slot {t}.")
                    valid = False

                # Vérification des incompatibilités
                if (task, operator) in self.incompatibilities:
                    errors.append(f"Opérateur {operator} assigné à une tâche incompatible {task} au slot {t}.")
                    valid = False

                # Vérification des sessions consécutives
                if task == current_task:
                    current_session_length += 1
                else:
                    if current_task != -1:
                        if current_session_length < self.min_session_slots:
                            errors.append(f"Session trop courte pour la tâche {current_task} par l'opérateur {operator} au slot {t - current_session_length} (seulement {current_session_length} slots).")
                            valid = False
                        if current_session_length > self.max_session_slots:
                            errors.append(f"Session trop longue pour la tâche {current_task} par l'opérateur {operator} au slot {t - current_session_length} ({current_session_length} slots).")
                            valid = False

                    current_task = task
                    current_session_length = 1 if task != -1 else 0

                # Comptabilisation des slots travaillés
                if task != -1:
                    task_counts[task] += 1
                    work_slots += 1

            # Vérification finale des sessions
            if current_task != -1:
                if current_session_length < self.min_session_slots:
                    errors.append(f"Session trop courte pour la tâche {current_task} par l'opérateur {operator} en fin de planning (seulement {current_session_length} slots).")
                    valid = False
                if current_session_length > self.max_session_slots:
                    errors.append(f"Session trop longue pour la tâche {current_task} par l'opérateur {operator} en fin de planning ({current_session_length} slots).")
                    valid = False

            # Vérification des slots maximum par tâche
            for task, count in task_counts.items():
                if count > self.max_task_slots:
                    errors.append(f"Tâche {task} assignée trop souvent ({count} slots) à l'opérateur {operator}.")
                    valid = False

            # Vérification des slots travaillés minimum
            if work_slots < self.min_work_slots:
                errors.append(f"Opérateur {operator} n'a travaillé que {work_slots} slots, en dessous du minimum requis ({self.min_work_slots}).")
                valid = False

        return valid, errors

    def simulate_task_flow(self, schedule: list[list[int]], log=False) -> list[float]:
        """Simule la production d'unité à partir d'un planning"""
        # Initialisation des flux pour chaque tâche
        task_units = [0] * self.n_tasks

        # Simulation sur chaque slot
        for t in range(self.time_slots):
            prod_task_units = [0] * self.n_tasks
            rema_task_units = task_units.copy() # Unités restantes, mises a jour a chaque op pour ne pas prendre plus d'unit que possible lors d'un slot

            if log: print(f"\n\n******** Time slot {t} *******\n");

            # Calcul des unités produites par tâche
            for operator in range(self.n_operators):

                task = schedule[operator][t]
                if log: print(f"### Operator {operator} on task {task}");

                if task != -1:
                    # Recupère les taches en amont de la tâche actuelle
                    previous_tasks = list(self.task_graph.predecessors(task))

                    # Calcul de la production max de la tache
                    production = self.task_capacity[task] * self.operators_prod_coef[operator][task]

                    if task != 0:
                        # Calcul de la production réelle
                        previous_tasks_units = [rema_task_units[p] * self.task_graph[p][task]['weight'] for p in previous_tasks]
                        unit_remainings = sum(previous_tasks_units)
                        production = min(unit_remainings, production)

                        # Ajout de la production réelle
                        prod_task_units[task] += production

                        if log and production != 0:
                            print(f"Added : {production} to {task}")

                        for p_task in previous_tasks:
                            # Calcul et déduction des unités consommés
                            substitute = min(rema_task_units[p_task] * self.task_graph[p_task][task]['weight'], production)
                            production -= substitute
                            rema_task_units[p_task] -= substitute

                            if log and substitute != 0:
                                print(f"Removed : {substitute} from {p_task}")

                            if production == 0:
                                break
                    else:
                        prod_task_units[task] += production

                        if log and production != 0:
                            print(f"Added : {production} to {task}")

            # Mise à jour des unités produites par tâche
            for task in range(len(task_units)):
                task_units[task] = rema_task_units[task] + prod_task_units[task]

            if log: print(task_units);

        return task_units

    def total_produced(self, schedule: list[list[int]]) -> float:
        """
        Calcul le nombre de documents produits à partir d'un planning.
        Renvoie -1 si le planning ne respecte pas les contraintes.
        """
        valid, errors = self.validate_schedule(schedule)
        if not valid:
            return -1
        return self.simulate_task_flow(schedule)[-1]


def validate_submission(df: pd.DataFrame) -> tuple[bool, list[list[str]]]:
    """Validate the CSV submission file for the Kaggle competition."""
    errors = []

    # Validate columns
    if set(df.columns) != {"id", "schedule"}:
        errors.append("CSV must have 'id' and 'schedule' columns.")
        return False, errors

    id_instances = [
        '4_2_24_0', '4_2_24_1', '4_2_48_0', '4_2_48_1', '4_2_96_0',
        '4_2_96_1', '4_2_240_0', '4_2_240_1', '10_3_24_0', '10_3_24_1',
        '10_3_48_0', '10_3_48_1', '10_3_96_0', '10_3_96_1', '10_3_240_0',
        '10_3_240_1', '20_5_48_0', '20_5_48_1', '20_5_96_0', '20_5_96_1',
        '20_5_240_0', '20_5_240_1', '48_16_96_0', '48_16_96_1', '48_16_240_0',
        '48_16_240_1', '75_40_240_0', '75_40_240_1', '100_50_240_0', '100_50_240_1'
    ]
    seen_ids = set()
    all_valid = True

    for i, row in enumerate(df.to_dict("records")):
        id_ = row['id']
        schedule = row['schedule']

        # Validate id
        if id_ not in id_instances:
            errors.append(f"Row {i}: Instance ID '{id_}' does not exists.")
            all_valid = False

        # Check for duplicate IDs
        if id_ in seen_ids:
            errors.append(f"Row {i}: Duplicate instance ID '{id_}'.")
            all_valid = False
        else:
            seen_ids.add(id_)

        # Validate schedule format
        try:
            sch = ast.literal_eval(schedule)
            if not isinstance(sch, list):
                errors.append(f"Row {i}: 'schedule' must be a list. Found {type(sch)}.")
                all_valid = False
            elif sch and not all(isinstance(row, list) for row in sch):
                errors.append(f"Row {i}: 'schedule' must be a list of lists.")
                all_valid = False

        except (ValueError, SyntaxError):
            errors.append(f"Row {i}: 'schedule' is not a valid list representation.")
            all_valid = False

    return all_valid, errors