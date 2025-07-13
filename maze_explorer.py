import sys
import os
import random
import math
from abc import ABC, abstractmethod
from collections import deque
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map

class Explorer(AbstAgent):
    def __init__(self, env, config_file, primary_direction, secondary_direction, width, height, resc, classifier=None):
        super().__init__(env, config_file)
        self.set_state(VS.ACTIVE)

        self.env = env
        self.width = width
        self.height = height
        self.resc = resc
        self.classifier = classifier  # <<< classificador treinado

        self.primary_direction = primary_direction
        self.secondary_direction = secondary_direction

        self.reduced_area = min(width, height) / 4
        self.base_area = self.reduced_area
        self.max_area = max(width, height)

        self.x = 0
        self.y = 0
        self.map = Map()
        self.victims = {}
        self.visited = set()
        self.return_points = set()
        self.return_path = {}
        self.path_it = 0
        self.exploration_flag = True
        self.worst_move_scenario = 0

        self.visited.add((self.x, self.y))
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def direction_priority(self, dx, dy):
        score = 0
        progress = (self.TLIM - self.get_rtime()) / self.TLIM

        if progress < 0.5:
            if self.primary_direction == "up": score -= dy
            elif self.primary_direction == "down": score += dy
            elif self.primary_direction == "left": score -= dx
            elif self.primary_direction == "right": score += dx
        else:
            if self.secondary_direction == "up": score -= dy
            elif self.secondary_direction == "down": score += dy
            elif self.secondary_direction == "left": score -= dx
            elif self.secondary_direction == "right": score += dx

        return score

    def within_radius(self, coord):
        q = self.primary_direction

        if q == "up" and (coord[0] < 0 or coord[1] > 0): return False
        if q == "left" and (coord[0] > 0 or coord[1] > 0): return False
        if q == "down" and (coord[0] > 0 or coord[1] < 0): return False
        if q == "right" and (coord[0] < 0 or coord[1] < 0): return False

        if abs(coord[0]) > self.reduced_area: return False
        if abs(coord[1]) > self.reduced_area: return False

        return True

    def look_around(self):
        obstacles = self.check_walls_and_lim()
        neighbors = []

        for direction in range(8):
            dx, dy = AbstAgent.AC_INCR[direction]
            new_pos = (self.x + dx, self.y + dy)
            if obstacles[direction] == VS.CLEAR and new_pos not in self.visited and self.within_radius(new_pos):
                self.return_points.add(new_pos)
                neighbors.append((dx, dy))

        neighbors.sort(key=lambda n: self.direction_priority(n[0], n[1]))
        return neighbors

    def update_coordinates(self, dx, dy):
        self.x += dx
        self.y += dy
        self.visited.add((self.x, self.y))

    def check_for_victims(self, dx, dy, rtime_bef, rtime_aft):
        seq_victim = self.check_for_victim()
        if seq_victim != VS.NO_VICTIM:
            vs = self.read_vital_signals()
            self.victims[vs[0]] = ((self.x, self.y), vs)

            if self.classifier:
                try:
                    input_features = [[vs[6], vs[7]]]  # <<< ajuste para o formato real!
                    prediction = self.classifier.predict(input_features)
                    print(f"{self.NAME} :: Victim at ({self.x},{self.y}) :: Predicted: {prediction}")
                except Exception as e:
                    print(f"{self.NAME} :: Classifier error: {e}")

        difficulty = (rtime_bef - rtime_aft)
        if dx == 0 or dy == 0:
            difficulty /= self.COST_LINE
        else:
            difficulty /= self.COST_DIAG

        self.map.add((self.x, self.y), difficulty, seq_victim, self.check_walls_and_lim())

    def get_neighbors(self, node, objective, has_objective):
        neighbors = []
        for direction in range(8):
            dx, dy = AbstAgent.AC_INCR[direction]
            coord = (node[0] + dx, node[1] + dy)
            if self.map.in_map(coord):
                neighbors.append(coord)
            if has_objective and objective == coord:
                neighbors.append(coord)
        return neighbors

    def compute_path_to_base(self, goal):
        if self.x == 0 and self.y == 0:
            return []

        start = (self.x, self.y)
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
                    path.append((dx, dy))
                    current = prev
                path.reverse()
                return path

            for neighbor in self.get_neighbors(current, goal, goal != (0, 0)):
                if neighbor not in visited:
                    visited.add(neighbor)
                    parent[neighbor] = current
                    queue.append(neighbor)
        return []

    def explore(self):
        neighbors = self.look_around()

        if neighbors:
            dx, dy = neighbors[0]
            rtime_bef = self.get_rtime()
            result = self.walk(dx, dy)
            rtime_aft = self.get_rtime()
            self.worst_move_scenario = max(self.worst_move_scenario, rtime_bef - rtime_aft)

            if result == VS.EXECUTED:
                self.update_coordinates(dx, dy)
                self.check_for_victims(dx, dy, rtime_bef, rtime_aft)
        else:
            self.backtrack()

    def backtrack(self):
        if self.return_points:
            point = self.return_points.pop()
            path = self.compute_path_to_base(point)
            for dx, dy in path:
                if self.walk(dx, dy) == VS.EXECUTED:
                    self.x += dx
                    self.y += dy
                    self.visited.add((self.x, self.y))
        else:
            self.exploration_flag = False

    def estimate_return_time(self):
        path = self.compute_path_to_base((0, 0))
        return path, len(path) * self.worst_move_scenario

    def can_explore(self):
        path, return_time = self.estimate_return_time()
        if return_time + 40 >= self.get_rtime():
            self.exploration_flag = False
        return path

    def returnto_base(self):
        if self.path_it < len(self.return_path):
            dx, dy = self.return_path[self.path_it]
            self.path_it += 1
            if self.walk(dx, dy) == VS.EXECUTED:
                self.x += dx
                self.y += dy

    def update_area(self):
        progress = (self.TLIM - self.get_rtime()) / self.TLIM
        growth_phase = 2.0 if progress >= 0.1 else progress / 0.1
        self.reduced_area = self.base_area + growth_phase * (self.max_area - self.base_area)

    def deliberate(self) -> bool:
        self.update_area()

        if self.exploration_flag:
            self.explore()
            self.return_path = self.can_explore()
        else:
            self.returnto_base()

        if self.x == 0 and self.y == 0:
            self.resc.sync_explorers(self.map, self.victims)
            return False

        return True
