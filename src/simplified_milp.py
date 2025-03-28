import pulp
import pandas as pd
import numpy as np
import time
from script import Instance

class SimplifiedMILPSolver:
    """
    Solveur MILP simplifié pour la planification des opérateurs.

    Ce solveur encode toutes les contraintes du problème, y compris
    les contraintes de flux instantané, mais omet les variables de sessions
    internes pour réduire la taille du modèle.

    Auteurs:
        Amine ROSTANE, Gaël GARNIER, Augustin LOGEAIS.
    """


    def __init__(self, instance, time_limit=None, mip_gap=0.01, verbose=True):
        """
        Initialise le solveur avec une instance de problème et des paramètres.

        Args:
            instance: L'instance du problème (au format compatible avec script.py).
            time_limit (int, optionnel): Temps maximal de résolution en secondes.
            mip_gap (float): Tolérance relative du gap MIP (ex. 0.01 pour 1%).
            verbose (bool): Active ou non l'affichage des logs.
        """

        self.instance = instance
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.verbose = verbose

        self.model = pulp.LpProblem("Simplified_Operator_Scheduling", pulp.LpMaximize)

        self.x = {}  
        self.y = {}  
        self.z = {}  
        self.task_production = {}   
        self.task_capacity_t = {}   
        self.leftover = {}          

        self.build_model()

    def build_model(self):
        """
        Construit le modèle MILP simplifié avec toutes les contraintes.

        Crée les variables de décision, la fonction objectif (production finale),
        et l’ensemble des contraintes nécessaires à la validité du planning.
        Supprime les variables de session pour améliorer les performances.
        """
        if self.verbose:
            print("Building simplified MILP model...")
            build_start = time.time()

        for o in range(self.instance.n_operators):
            for t in range(self.instance.time_slots):

                self.y[o, t] = pulp.LpVariable(f"y_{o}_{t}", cat=pulp.LpBinary)

                for m in range(self.instance.n_tasks):
                    self.x[o, t, m] = pulp.LpVariable(f"x_{o}_{t}_{m}", cat=pulp.LpBinary)

        for o in range(self.instance.n_operators):
            for t in range(self.instance.time_slots):
                for m in range(self.instance.n_tasks):
                    self.z[o, t, m] = pulp.LpVariable(f"z_{o}_{t}_{m}", cat=pulp.LpBinary)

        for m in range(self.instance.n_tasks):
            for t in range(self.instance.time_slots):
                self.task_production[m, t] = pulp.LpVariable(f"prod_{m}_{t}", lowBound=0)
                self.task_capacity_t[m, t] = pulp.LpVariable(f"cap_{m}_{t}", lowBound=0)

        for m in range(self.instance.n_tasks):
            for t in range(self.instance.time_slots + 1):
                self.leftover[m, t] = pulp.LpVariable(f"leftover_{m}_{t}", lowBound=0)

        last_task = self.instance.n_tasks - 1
        self.model += pulp.lpSum(
            self.task_production[last_task, t] for t in range(self.instance.time_slots)
        ), "MaximizeFinalProduction"

        for o in range(self.instance.n_operators):
            start_avail = self.instance.operators_availability[o]['start']
            end_avail = self.instance.operators_availability[o]['end']
            for t in range(self.instance.time_slots):

                self.model += (
                    pulp.lpSum([self.x[o, t, m] for m in range(self.instance.n_tasks)])
                    + self.y[o, t] == 1
                )

                if t < start_avail or t > end_avail:
                    self.model += self.y[o, t] == 1  

        for (task_incomp, operator_incomp) in self.instance.incompatibilities:
            for t in range(self.instance.time_slots):
                self.model += self.x[operator_incomp, t, task_incomp] == 0

        for m in range(self.instance.n_tasks):
            base_cap = self.instance.task_capacity[m]
            for t in range(self.instance.time_slots):
                self.model += (
                    self.task_capacity_t[m, t]
                    == base_cap * pulp.lpSum(
                        self.x[o, t, m] * self.instance.operators_prod_coef[o][m]
                        for o in range(self.instance.n_operators)
                    )
                )

        for o in range(self.instance.n_operators):
            for m in range(self.instance.n_tasks):
                for t in range(self.instance.time_slots):
                    if t == 0:
                        self.model += self.z[o, t, m] == self.x[o, t, m]
                    else:
                        self.model += self.z[o, t, m] >= self.x[o, t, m] - self.x[o, t-1, m]
                        self.model += self.z[o, t, m] <= self.x[o, t, m]
                        self.model += self.z[o, t, m] <= 1 - self.x[o, t-1, m]

        min_slots = self.instance.min_session_slots

        for o in range(self.instance.n_operators):
            for m in range(self.instance.n_tasks):
                for t in range(self.instance.time_slots - min_slots + 1):
                    for i in range(1, min_slots):
                        self.model += self.x[o, t+i, m] >= self.z[o, t, m]

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
            self.model += (
                pulp.lpSum(
                    self.x[o, t, m]
                    for m in range(self.instance.n_tasks)
                    for t in range(self.instance.time_slots)
                ) >= self.instance.min_work_slots
            )

        bigM = 1000000
        for m in range(self.instance.n_tasks):
            if m == 0:
                self.model += self.leftover[m, 0] == bigM  
            else:
                self.model += self.leftover[m, 0] == 0

        for t in range(self.instance.time_slots):
            for m in range(self.instance.n_tasks):
                if m == 0:
                    supply_expr = self.leftover[m, t]
                else:
                    preds = list(self.instance.task_graph.predecessors(m))
                    supply_expr = (
                        self.leftover[m, t] +
                        pulp.lpSum(
                            self.task_production[pred, t] * self.instance.task_graph[pred][m]['weight']
                            for pred in preds
                        )
                    )

                self.model += self.task_production[m, t] <= self.task_capacity_t[m, t]
                self.model += self.task_production[m, t] <= supply_expr

                if t < self.instance.time_slots:
                    if m == 0:

                        self.model += (
                            self.leftover[m, t+1]
                            == self.leftover[m, t] - self.task_production[m, t]
                        )
                    else:
                        preds = list(self.instance.task_graph.predecessors(m))
                        self.model += (
                            self.leftover[m, t+1]
                            == self.leftover[m, t]
                            + pulp.lpSum(
                                self.task_production[pred, t] * self.instance.task_graph[pred][m]['weight']
                                for pred in preds
                            )
                            - self.task_production[m, t]
                        )

        if self.verbose:
            n_vars = len(self.model.variables())
            n_cons = len(self.model.constraints)
            build_time = time.time() - build_start
            print(f"Model built with {n_vars} variables and {n_cons} constraints "
                  f"in {build_time:.2f} seconds.")

    def solve(self):
        """
        Résout le modèle MILP et extrait le meilleur planning trouvé.

        Returns:
            tuple: Un tuple (schedule, score) où :
                - schedule (List[List[int]]): Matrice O*T représentant le planning trouvé.
                - score (float): Valeur de la fonction objectif (production totale).
        """

        if self.verbose:
            print(f"Solving model with time limit={self.time_limit} sec, MIP gap={self.mip_gap}...")

        if self.time_limit is not None:
            solver = pulp.PULP_CBC_CMD(
                timeLimit=self.time_limit,
                gapRel=self.mip_gap,
                msg=self.verbose
            )
        else:
            solver = pulp.PULP_CBC_CMD(gapRel=self.mip_gap, msg=self.verbose)

        start_solve = time.time()
        self.model.solve(solver)
        solve_duration = time.time() - start_solve
        status = pulp.LpStatus[self.model.status]

        if self.verbose:
            print(f"Solve ended with status={status} in {solve_duration:.2f} s")

        if status not in ["Optimal", "Feasible"]:
            print("No feasible solution found. Returning None.")
            return None, -1

        schedule = [[-1]*self.instance.time_slots for _ in range(self.instance.n_operators)]
        for o in range(self.instance.n_operators):
            for t in range(self.instance.time_slots):
                for m in range(self.instance.n_tasks):
                    if pulp.value(self.x[o, t, m]) > 0.5:
                        schedule[o][t] = m

        objective_value = pulp.value(self.model.objective)
        return schedule, objective_value

def main(instance_path, time_limit=600, mip_gap=0.01, verbose=True, save_csv=True):
    """Exécute le solveur MILP sur une instance et affiche les résultats.

    Args:
        instance_path (str): Chemin du fichier JSON de l’instance à résoudre.
        time_limit (int): Temps limite pour la résolution.
        mip_gap (float): Tolérance relative du gap MIP.
        verbose (bool): Affichage détaillé des étapes du solveur.
        save_csv (bool): Sauvegarder ou non le planning dans un fichier CSV.
    """

    print(f"\n🔹 Loading instance: {instance_path} ...")
    instance = Instance(instance_path)

    print(f"🔹 Instance Details: {instance.n_operators} operators, {instance.n_tasks} tasks, {instance.time_slots} time slots.")
    print(f"   ➤ Min session: {instance.min_session_slots} slots, Max session: {instance.max_session_slots} slots.")
    print(f"   ➤ Operator shift range: {min(op['start'] for op in instance.operators_availability)} - {max(op['end'] for op in instance.operators_availability)}.")
    print(f"   ➤ Constraints: Max task slots: {instance.max_task_slots}, Min work slots: {instance.min_work_slots}.")

    print("\n🔹 Initializing MILP Solver...")
    solver = SimplifiedMILPSolver(instance, time_limit=time_limit, mip_gap=mip_gap, verbose=verbose)

    print("\n🔹 Solving MILP...")
    start_time = time.time()
    schedule, obj_val = solver.solve()
    solve_time = time.time() - start_time

    if schedule is None:
        print("\n❌ No valid solution found. Exiting.")
        return

    print(f"\n✅ Solution Found! Solver Status: {pulp.LpStatus[solver.model.status]}")
    print(f"   ➤ Objective Value (Solver-reported final task production): {obj_val}")
    print(f"   ➤ Solve Time: {solve_time:.2f} sec")

    print("\n🔹 Validating Schedule with script.py...")
    valid, errors = instance.validate_schedule(schedule)

    if valid:
        actual_production = instance.total_produced(schedule)
        print(f"✅ Schedule is valid! Total Production (from script.py flow simulation): {actual_production}")
        if abs(actual_production - obj_val) > 1e-6:
            print(f"⚠️ Warning: Production mismatch! Solver: {obj_val}, Script.py: {actual_production}")
    else:
        print("❌ Invalid Schedule! Errors:")
        for err in errors:
            print("   ➤", err)
        return

    print("\n🔹 Schedule Summary:")
    for o in range(min(5, instance.n_operators)):  
        print(f"   ➤ Operator {o}: {schedule[o][:min(30, instance.time_slots)]} ...")

    if save_csv:
        print("\n🔹 Saving schedule to 'submission.csv'...")
        df = pd.DataFrame({"id": [instance_path.split("/")[-1]], "schedule": [schedule]})
        df.to_csv("submission.csv", index=False)
        print("✅ CSV saved successfully!")

    print("\n🎯 Optimization Completed! 🎯")

if __name__ == "__main__":
    main("4_2_24_0.json", time_limit=600, mip_gap=0.01, verbose=True)