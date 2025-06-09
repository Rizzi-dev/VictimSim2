import sys
import os
import random
import math
from abc import ABC, abstractmethod
from vs.abstract_agent import AbstAgent
from vs.constants import VS
from map import Map

class Stack:
    def __init__(self):
        self.items = []

    def push(self, item):
        self.items.append(item)

    def pop(self):
        if not self.is_empty():
            return self.items.pop()

    def is_empty(self):
        return len(self.items) == 0

class Explorer(AbstAgent):
    def __init__(self, env, config_file, resc, initial_d):
        super().__init__(env, config_file)
        self.walk_stack = Stack()
        self.set_state(VS.ACTIVE)
        self.resc = resc
        self.start = True
        self.initial_direction = initial_d
        self.x = 0
        self.y = 0
        self.map = Map()
        self.victims = {}
        self.visited = set()
        self.visited.add((self.x, self.y))
        self.map.add((self.x, self.y), 1, VS.NO_VICTIM, self.check_walls_and_lim())

    def get_unvisited_neighbors(self):
        obstacles = self.check_walls_and_lim()
        neighbors = []

        for direction in range(8):
            dx, dy = AbstAgent.AC_INCR[direction]
            new_pos = (self.x + dx, self.y + dy)
            if obstacles[direction] == VS.CLEAR and new_pos not in self.visited:
                neighbors.append((direction, dx, dy))

        neighbors.sort(key=lambda tup: (tup[0] - self.initial_direction) % 8)

        return [(dx, dy) for _, dx, dy in neighbors]



    def explore(self):
        neighbors = self.get_unvisited_neighbors()

        if neighbors:
            dx, dy = neighbors[0]
            rtime_bef = self.get_rtime()
            result = self.walk(dx, dy)
            rtime_aft = self.get_rtime()

            if result == VS.EXECUTED:
                self.walk_stack.push((dx, dy))
                self.x += dx
                self.y += dy
                self.visited.add((self.x, self.y))

                seq = self.check_for_victim()
                if seq != VS.NO_VICTIM:
                    vs = self.read_vital_signals()
                    self.victims[vs[0]] = ((self.x, self.y), vs)
                    print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")

                difficulty = (rtime_bef - rtime_aft)
                if dx == 0 or dy == 0:
                    difficulty = difficulty / self.COST_LINE
                else:
                    difficulty = difficulty / self.COST_DIAG

                self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())

        else:
            self.come_back()  # no unvisited neighbor → backtrack
            
       
    def come_back(self):
        if self.walk_stack.is_empty():
            return
        dx, dy = self.walk_stack.pop()
        dx = -dx
        dy = -dy
        result = self.walk(dx, dy)
        if result == VS.EXECUTED:
            self.x += dx
            self.y += dy 
            
    def start_in_direction(self):
        dx, dy = AbstAgent.AC_INCR[self.initial_direction ]
        result = self.walk(dx, dy)
        if result == VS.EXECUTED:
            self.walk_stack.push((dx, dy))
            self.x += dx
            self.y += dy
            self.visited.add((self.x, self.y))

            seq = self.check_for_victim()
            if seq != VS.NO_VICTIM:
                vs = self.read_vital_signals()
                self.victims[vs[0]] = ((self.x, self.y), vs)
                print(f"{self.NAME} Victim found at ({self.x}, {self.y}), rtime: {self.get_rtime()}")

            rtime_bef = self.get_rtime()
            difficulty = rtime_bef  # first move, no previous rtime to compare

            if dx == 0 or dy == 0:
                difficulty = difficulty / self.COST_LINE
            else:
                difficulty = difficulty / self.COST_DIAG

            self.map.add((self.x, self.y), difficulty, seq, self.check_walls_and_lim())
        else:
            print(f"{self.NAME}: Failed to start in direction {direction}, obstacle or invalid move.")

            

        
    def deliberate(self) -> bool:
        if(self.start):
            self.start_in_direction()
            self.start = False
        consumed_time = self.TLIM - self.get_rtime()
        # print(f"{self.NAME}: consumed time {consumed_time}, rtime {self.get_rtime()}")
        if consumed_time < self.get_rtime():
            self.explore()
            return True

        if self.walk_stack.is_empty() or (self.x == 0 and self.y == 0):
            print(f"{self.NAME}: rtime {self.get_rtime()}, invoking the rescuer")
            self.resc.go_save_victims(self.map, self.victims)
            return False

        self.come_back()
        return True
        


        