import os
import random
from map import Map
from vs.abstract_agent import AbstAgent
from vs.physical_agent import PhysAgent
from vs.constants import VS
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from victims_sequencer import Sequencer
from collections import deque
import numpy as np
import csv
import time

# 👇 Importa os modelos do seu arquivo de treinamento
from classifier_regressor import train_test_regressor, train_test_classifier

class Rescuer(AbstAgent):
    # 👇 Modelos estáticos (treinam 1x só)
    regressor = train_test_regressor("CART")   # Ou "MLP"
    classifier = train_test_classifier("CART") # Ou "MLP"

    def __init__(self, env, config_file, nb_of_explorers, cluster=[]):
        super().__init__(env, config_file)
        self.nb_of_explorers = nb_of_explorers
        self.received_maps = 0
        self.map = Map() 
        self.walk_constant = 1.75
        self.all_victims = {} 
        self.plan = []              
        self.plan_x = 0             
        self.plan_y = 0             
        self.plan_visited = set()   
        self.plan_rtime = self.TLIM 
        self.plan_walk_time = 0.0   
        self.cluster = cluster
        self.rescue_plan = {}
        self.x = 0                  
        self.y = 0                  
        self.set_state(VS.IDLE)

    def predict_gravity(self, vital_signals):
        # vital_signals: (id, x, y, qPA, pulso, fResp)
        x_input = np.array(vital_signals[3:]).reshape(1, -1)
        return self.regressor.predict(x_input)[0]

    def predict_class(self, vital_signals):
        x_input = np.array(vital_signals[3:]).reshape(1, -1)
        return self.classifier.predict(x_input)[0]


    def save_predictions(self):
        with open("file_predict.txt", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "gravity", "class"])  # Cabeçalho opcional
            for victim_id, (coord, vs) in self.all_victims.items():
                gravity = self.predict_gravity(vs)
                label = self.predict_class(vs)
                writer.writerow([victim_id, coord[0], coord[1], gravity, label])

    def cluster_victims(self, victims_positions, n_clusters=4, random_state=42):
        X = np.array(victims_positions)
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(X)
        centroids = kmeans.cluster_centers_
        return labels, centroids

    def sync_explorers(self, explorer_map, victims):
        self.received_maps += 1
        self.map.update(explorer_map)
        self.all_victims.update(victims)

        if self.received_maps == self.nb_of_explorers:
            victims_positions = [coord for coord, _ in self.all_victims.values()]
            clusters = self.divide_victims(victims_positions)

            rescuers = [None] * 4
            rescuers[0] = self 

            self.cluster = [clusters[0]]

            for i in range(1, 4):
                filename = f"rescuer_{i+1:1d}_config.txt"
                config_file = os.path.join(self.config_folder, filename)
                rescuers[i] = Rescuer(self.get_env(), config_file, 4, [clusters[i]]) 
                rescuers[i].map = self.map    

            # 🔑 Gera e salva as predições
            self.save_predictions()

            for i, rescuer in enumerate(rescuers):
                rescuer.victims_rescue_seq()
                rescuer.planner()
                rescuer.set_state(VS.ACTIVE)

    def divide_victims(self, victims_positions):
        labels, _ = self.cluster_victims(victims_positions)
        victims_group = list(zip(victims_positions, labels.tolist()))
        clusters = {}
        for pos, cluster_id in victims_group:
            clusters.setdefault(cluster_id, []).append(pos)
        return clusters

    def victims_rescue_seq(self):
        rescue_plan = {}
        
        for _, victims in enumerate(self.cluster):
            if len(victims) == 1:
                rescue_plan = victims  
            else:
                best_route, _ = Sequencer().genetic_algorithm(victims)
                rescue_plan = best_route
            
        self.rescue_plan = rescue_plan

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
                if goal[0] != 0 and goal[1] != 0:
                    dx, dy, has_victim = path[-1]
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
            total_time += len(new_plan)*self.walk_constant
            prev_goal = coord

        return_plan = self.calculatepath_tovictim(prev_goal, (0, 0))
        
        while total_time + len(return_plan)*self.walk_constant >= self.TLIM and len(all_plans) > 0:
            total_time -= len(all_plans.pop())*self.walk_constant
            it = len(all_plans) - 1
            return_plan = self.calculatepath_tovictim(self.rescue_plan[it], (0, 0))
            
        if len(all_plans) > 0:
            all_plans.append(return_plan)
        
        for plan in all_plans:
            self.plan.extend(plan)

    def deliberate(self) -> bool:
        if self.plan == []:
           return False

        dx, dy, there_is_vict = self.plan.pop(0)
        walked = self.walk(dx, dy)

        if walked == VS.EXECUTED:
            self.x += dx
            self.y += dy
            
            if there_is_vict:
                rescued = self.first_aid() 
                if rescued:
                    print(f"{self.NAME} Victim rescued at ({self.x}, {self.y})")
                else:
                    print(f"{self.NAME} Plan fail - victim not found at ({self.x}, {self.y})")
        else:
            print(f"{self.NAME} Plan fail - walk error - agent at ({self.x}, {self.y})")

        return True
