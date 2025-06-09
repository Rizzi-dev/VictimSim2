# agents/ExplorerAgent.py (Versão com Depuração no __init__)

import math
from queue import PriorityQueue
from vs.abstract_agent import AbstAgent

class Explorer(AbstAgent):
    def __init__(self, env, config_file):
        # --- PRINTS DE DEBUB NO CONSTRUTOR ---
        print("--- [AGENTE] Iniciando o construtor __init__ do ExplorerAgent ---")
        
        print("--- [AGENTE] PASSO 1: Chamando super().__init__(...) para registrar no ambiente.")
        super().__init__(env, config_file)
        print("--- [AGENTE] PASSO 2: super().__init__(...) foi concluído.")
        
        self.map = {(0, 0): 'base'}
        self.unexplored_frontier = set()
        self.victims_found = {}
        self.plan = []

        self.current_pos = (0, 0)
        self.base_pos = (0, 0)
        self.battery = self.TLIM
        self.time_spent = 0.0

        self.cost_line = self.COST_LINE
        self.cost_diag = self.COST_DIAG
        self.cost_read = self.COST_READ
        
        self.delta_to_action = {
            (0, -1): 'move_north', (0, 1): 'move_south', (1, 0): 'move_east', (-1, 0): 'move_west',
            (1, -1): 'move_north_east', (-1, -1): 'move_north_west',
            (1, 1): 'move_south_east', (-1, 1): 'move_south_west'
        }
        self.deliberate_count = 0
        print("--- [AGENTE] PASSO 3: Atributos internos inicializados.")
        print("--- [AGENTE] Construtor __init__ do ExplorerAgent finalizado com sucesso! ---")


    def deliberate(self) -> str:
        self.deliberate_count += 1
        print(f"\n--- Ciclo Deliberate #{self.deliberate_count} --- Posição: {self.current_pos}")
        # ... o resto do código do deliberate continua o mesmo
        
        self.perceive_and_update_map()
        # ... (código omitido para brevidade, ele permanece o mesmo da versão anterior)
        if not self.plan:
            if self.must_return_to_base():
                path_to_base = self.astar_path(self.current_pos, self.base_pos)
                if path_to_base: self.plan = self.convert_path_to_actions(path_to_base)
            else:
                goal = self.find_best_unexplored_cell()
                if goal:
                    self.unexplored_frontier.remove(goal)
                    path_to_goal = self.astar_path(self.current_pos, goal)
                    if path_to_goal: self.plan = self.convert_path_to_actions(path_to_goal)
        if self.plan:
            action = self.plan.pop(0)
            sensed_data = self.body.sense()
            if sensed_data and sensed_data['victim'] and not sensed_data['victim_read']:
                if action != 'read_vital_signs': self.plan.insert(0, action)
                action = 'read_vital_signs'
            self.update_agent_state(action)
            return action
        return "no-op"
        
    # Todas as outras funções (perceive_and_update_map, astar_path, etc.) permanecem as mesmas
    def perceive_and_update_map(self):
        sensed_data = self.body.sense()
        if not sensed_data: return
        if sensed_data['victim'] and not sensed_data['victim_read']:
            self.map[self.current_pos] = 'victim'
        elif self.map.get(self.current_pos) != 'base':
             self.map[self.current_pos] = 'empty'
        for direction, cell_data in sensed_data['adjacent'].items():
            dx, dy = self.get_delta_from_direction(direction)
            adj_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
            if adj_pos not in self.map:
                if cell_data['type'] == 'obstacle': self.map[adj_pos] = 'obstacle'
                else:
                    self.map[adj_pos] = 'unexplored'
                    self.unexplored_frontier.add(adj_pos)
    def must_return_to_base(self) -> bool:
        battery_left = self.battery - self.time_spent
        estimated_cost = self.heuristic(self.current_pos, self.base_pos) * self.cost_line
        if battery_left < estimated_cost: return True
        if battery_left < (estimated_cost * 1.5):
            path_back = self.astar_path(self.current_pos, self.base_pos)
            if not path_back: return True
            cost_to_return = self.calculate_path_cost(path_back)
            return battery_left < (cost_to_return * 1.15)
        return False
    def find_best_unexplored_cell(self):
        if not self.unexplored_frontier: return None
        min_dist = float('inf')
        best_cell = None
        for cell in self.unexplored_frontier:
            dist = self.heuristic(self.current_pos, cell)
            if dist < min_dist:
                min_dist = dist
                best_cell = cell
        return best_cell
    def astar_path(self, start, goal):
        frontier = PriorityQueue(); frontier.put((0, start)); came_from = {start: None}; cost_so_far = {start: 0}
        while not frontier.empty():
            _, current = frontier.get()
            if current == goal: break
            for neighbor, move_cost in self.get_neighbors(current):
                new_cost = cost_so_far[current] + move_cost
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self.heuristic(goal, neighbor)
                    frontier.put((priority, neighbor)); came_from[neighbor] = current
        if goal not in came_from: return None
        path = []; current = goal
        while current is not None: path.append(current); current = came_from[current]
        path.reverse(); return path
    def get_neighbors(self, pos):
        (x, y) = pos; neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0: continue
                neighbor_pos = (x + dx, y + dy)
                if self.map.get(neighbor_pos, 'obstacle') != 'obstacle':
                    move_cost = self.cost_line if dx == 0 or dy == 0 else self.cost_diag
                    neighbors.append((neighbor_pos, move_cost))
        return neighbors
    def heuristic(self, a, b):
        (x1, y1) = a; (x2, y2) = b; return abs(x1 - x2) + abs(y1 - y2)
    def convert_path_to_actions(self, path):
        actions = []
        for i in range(len(path) - 1):
            curr_pos, next_pos = path[i], path[i+1]
            dx, dy = next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1]
            if (dx, dy) in self.delta_to_action: actions.append(self.delta_to_action[(dx, dy)])
        return actions
    def update_agent_state(self, action):
        action_cost = 0; dx, dy = 0, 0
        if action.startswith('move'):
            action_cost = self.cost_diag if '_' in action else self.cost_line
            if 'north' in action: dy = -1
            if 'south' in action: dy = 1
            if 'east' in action: dx = 1
            if 'west' in action: dx = -1
        elif action == 'read_vital_signs':
            action_cost = self.cost_read
            sensed_data = self.body.sense()
            if sensed_data and sensed_data['victim']:
                self.victims_found[self.current_pos] = sensed_data['victim_info']
        self.current_pos = (self.current_pos[0] + dx, self.current_pos[1] + dy)
        self.time_spent += action_cost
    def calculate_path_cost(self, path):
        cost = 0
        for i in range(len(path) - 1):
            dx, dy = path[i+1][0] - path[i][0], path[i+1][1] - path[i][1]
            cost += self.cost_line if dx == 0 or dy == 0 else self.cost_diag
        return cost
    def get_delta_from_direction(self, direction: str):
        dy = -1 if 'north' in direction else 1 if 'south' in direction else 0
        dx = 1 if 'east' in direction else -1 if 'west' in direction else 0
        return dx, dy