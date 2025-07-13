import os
import random
from map import Map
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from victims_sequencer import Sequencer
from collections import deque
import numpy as np
import csv

from classifier_regressor import train_test_regressor, train_test_classifier

class Rescuer(AbstAgent):
    # Modelos fixos
    regressor = train_test_regressor("CART")
    classifier = train_test_classifier("CART")

    def __init__(self, env, config_file, nb_of_explorers, cluster=[]):
        super().__init__(env, config_file)
        self.nb_of_explorers = nb_of_explorers
        self.received_maps = 0
        self.map = Map()
        self.walk_constant = 1.75
        self.all_victims = {}
        self.plan = []
        self.cluster = cluster
        self.rescue_plan = {}
        self.x = 0
        self.y = 0
        self.set_state(VS.IDLE)

    def predict_gravity(self, vs):
        x_input = np.array(vs[3:]).reshape(1, -1)
        return self.regressor.predict(x_input)[0]

    def predict_class(self, vs):
        x_input = np.array(vs[3:]).reshape(1, -1)
        return self.classifier.predict(x_input)[0]

    def save_predictions(self):
        with open("file_predict.txt", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "gravity", "class"])
            for victim_id, (coord, vs) in self.all_victims.items():
                g = self.predict_gravity(vs)
                c = self.predict_class(vs)
                writer.writerow([victim_id, coord[0], coord[1], g, c])

    def cluster_victims(self, victims_info, n_clusters=4):
        X = np.array([info["coord"] for info in victims_info])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(X)
        return labels

    def save_clusters(self, clusters):
        with open("file_clusters.txt", "w") as f:
            for cluster_id, victims in clusters.items():
                f.write(f"Cluster {cluster_id}:\n")
                for v in victims:
                    f.write(f"  {v}\n")
                f.write("\n")

    def save_rescue_plan(self):
        with open(f"file_rescue_plan_{self.NAME}.txt", "w") as f:
            for step in self.plan:
                dx, dy, has_victim = step
                f.write(f"dx={dx}, dy={dy}, victim={has_victim}\n")

    def sync_explorers(self, explorer_map, victims):
        self.received_maps += 1
        self.map.update(explorer_map)
        self.all_victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            victims_info = []
            for _, (coord, vs) in self.all_victims.items():
                g = self.predict_gravity(vs)
                c = self.predict_class(vs)
                victims_info.append({"coord": coord, "gravity": g, "class": c, "vs": vs})

            # Salva predições
            self.save_predictions()

            labels = self.cluster_victims(victims_info)
            clusters = {}
            for info, label in zip(victims_info, labels):
                clusters.setdefault(label, []).append(info)

            # Ordena cada cluster por gravidade decrescente
            for cluster_id in clusters:
                clusters[cluster_id].sort(key=lambda v: v["gravity"], reverse=True)

            self.save_clusters(clusters)

            rescuers = [None] * 4
            rescuers[0] = self
            self.cluster = [clusters[0]]

            for i in range(1, 4):
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, [clusters[i]])
                rescuers[i].map = self.map

            for i, rescuer in enumerate(rescuers):
                rescuer.victims_rescue_seq()
                rescuer.planner()
                rescuer.set_state(VS.ACTIVE)
                rescuer.save_rescue_plan()

    def victims_rescue_seq(self):
        sequence = []
        for cluster in self.cluster:
            if len(cluster) == 1:
                sequence.append(cluster[0]["coord"])
            else:
                coords = [v["coord"] for v in cluster]
                best_route, _ = Sequencer().genetic_algorithm(coords)
                sequence.extend(best_route)
        self.rescue_plan = sequence

    def get_neighbors(self, node):
        neighbors = []
        for direction in range(8):
            dx, dy = AbstAgent.AC_INCR[direction]
            coord = (node[0] + dx, node[1] + dy)
            if self.map.in_map(coord):
                neighbors.append(coord)
        return neighbors

    def calculatepath_tovictim(self, start, goal):
        queue = deque([start])
        visited = set([start])
        parent = {}

        while queue:
            current = queue.popleft()
            if current == goal:
                path = []
                while current != start:
                    prev = parent[current]
                    dx = current[0] - prev[0]
                    dy = current[1] - prev[1]
                    path.append((dx, dy, False))
                    current = prev
                path.reverse()
                if goal != (0, 0):
                    dx, dy, _ = path[-1]
                    path[-1] = (dx, dy, True)
                return path

            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
        return {}

    def planner(self):
        prev_goal = (0, 0)
        all_plans = []
        total_time = 0

        for coord in self.rescue_plan:
            new_plan = self.calculatepath_tovictim(prev_goal, coord)
            all_plans.append(new_plan)
            total_time += len(new_plan) * self.walk_constant
            prev_goal = coord

        return_plan = self.calculatepath_tovictim(prev_goal, (0, 0))

        while total_time + len(return_plan) * self.walk_constant >= self.TLIM and len(all_plans) > 0:
            total_time -= len(all_plans.pop()) * self.walk_constant
            prev_goal = (0, 0) if len(all_plans) == 0 else self.rescue_plan[len(all_plans) - 1]
            return_plan = self.calculatepath_tovictim(prev_goal, (0, 0))

        if len(all_plans) > 0:
            all_plans.append(return_plan)

        for plan in all_plans:
            self.plan.extend(plan)

    def deliberate(self) -> bool:
        if not self.plan:
            return False

        dx, dy, there_is_vict = self.plan.pop(0)
        walked = self.walk(dx, dy)

        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy

            if there_is_vict:
                if self.first_aid():
                    print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
                else:
                    print(f"{self.NAME} Victim NOT found at ({self.x}, {self.y})")
        else:
            print(f"{self.NAME} Walk fail at ({self.x}, {self.y})")

        return True
