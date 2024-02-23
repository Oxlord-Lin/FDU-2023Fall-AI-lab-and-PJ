# -*- encoding: utf-8 -*-
# Author: Jiwen Zhang
# Time: 04/08 23:33
# Version: 2.0
# Contact: jwzhang16@fudan.edu.cn
# Reference: MattChanTK@github

import os
import time
import random


class Maze:

    ACTIONS = ["U", "R", "D", "L"]
    COMPASS = {
        "U":  (0, -1),  # turn up
        "R":  (1, 0),   # turn right
        "D":  (0, 1),   # turn down
        "L":  (-1, 0)   # turn left
    }

    def __init__(self, maze_size=(10,10), has_loops=True, num_portals=0):
        """ Initialize the maze. """

        if not (isinstance(maze_size, (list, tuple)) and len(maze_size) == 2):
            raise ValueError("maze_size must be a tuple: (width, height).")
        self.maze_size = maze_size

        self.has_loops = has_loops
        self.__portals_dict = dict()
        self.__portals = []
        self.num_portals = num_portals

        self.maze_cells = None

        self._generate_maze()

    def _generate_maze(self):

        # list of all cell locations
        self.maze_cells = [[0]*self.maze_size[0] for j in range(self.maze_size[1])]

        # Initializing constants and variables needed for maze generation
        current_cell = (random.randint(0, self.MAZE_W-1), random.randint(0, self.MAZE_H-1))
        num_cells_visited = 1
        cell_stack = [current_cell]

        # Continue until all cells are visited
        while cell_stack:

            # restart from a cell from the cell stack
            current_cell = cell_stack.pop()
            x0, y0 = current_cell

            # find neighbours of the current cells that actually exist
            neighbours = dict()
            for dir_key, dir_val in self.COMPASS.items():
                x1 = x0 + dir_val[0]
                y1 = y0 + dir_val[1]
                # if cell is within bounds
                if 0 <= x1 < self.MAZE_W and 0 <= y1 < self.MAZE_H:
                    # if all four walls still exist
                    if self.all_walls_intact(self.maze_cells[x1][y1]):
                        neighbours[dir_key] = (x1, y1)

            # if there is a neighbour
            if neighbours:
                # select a random neighbour
                dir = random.choice(tuple(neighbours.keys()))
                x1, y1 = neighbours[dir]

                # knock down the wall between the current cell and the selected neighbour
                self.maze_cells[x1][y1] = self.__break_walls(self.maze_cells[x1][y1], self.__get_opposite_wall(dir))

                # push the current cell location to the stack
                cell_stack.append(current_cell)

                # make the this neighbour cell the current cell
                cell_stack.append((x1, y1))

                # increment the visited cell count
                num_cells_visited += 1

        if self.has_loops:
            self.__break_random_walls(0.2)

        if self.num_portals > 0:
            self.__set_random_portals(num_portal_sets=self.num_portals, set_size=2)

    def __break_random_walls(self, percent):
        # find some random cells to break
        num_cells = int(round(self.MAZE_H*self.MAZE_W*percent))
        cell_ids = random.sample(range(self.MAZE_W*self.MAZE_H), num_cells)

        # for each of those walls
        for cell_id in cell_ids:
            x = cell_id % self.MAZE_H
            y = int(cell_id/self.MAZE_H)

            # randomize the compass order
            dirs = random.sample(list(self.COMPASS.keys()), len(self.COMPASS))
            for dir in dirs:
                # break the wall if it's not already open
                if self.is_breakable((x, y), dir):
                    self.maze_cells[x][y] = self.__break_walls(self.maze_cells[x][y], dir)
                    break

    def __set_random_portals(self, num_portal_sets, set_size=2):
        # find some random cells to break
        num_portal_sets = int(num_portal_sets)
        set_size = int(set_size)

        # limit the maximum number of portal sets to the number of cells available.
        max_portal_sets = int(self.MAZE_W * self.MAZE_H / set_size)
        num_portal_sets = min(max_portal_sets, num_portal_sets)

        # the first and last cells are reserved
        cell_ids = random.sample(range(1, self.MAZE_W * self.MAZE_H - 1), num_portal_sets*set_size)

        for i in range(num_portal_sets):
            # sample the set_size number of sell
            portal_cell_ids = random.sample(cell_ids, set_size)
            portal_locations = []
            for portal_cell_id in portal_cell_ids:
                # remove the cell from the set of potential cell_ids
                cell_ids.pop(cell_ids.index(portal_cell_id))
                # convert portal ids to location
                x = portal_cell_id % self.MAZE_H
                y = int(portal_cell_id / self.MAZE_H)
                portal_locations.append((x,y))
            # append the new portal to the maze
            portal = Portal(*portal_locations)
            self.__portals.append(portal)

            # create a dictionary of portals
            for portal_location in portal_locations:
                self.__portals_dict[portal_location] = portal

    def is_open(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        # if cell is still within bounds after the move
        if self.is_within_bound(x1, y1):
            # check if the wall is opened
            this_wall = bool(self.get_walls_status(self.maze_cells[cell_id[0]][cell_id[1]])[dir])
            other_wall = bool(self.get_walls_status(self.maze_cells[x1][y1])[self.__get_opposite_wall(dir)])
            return this_wall or other_wall
        return False

    def is_breakable(self, cell_id, dir):
        # check if it would be out-of-bound
        x1 = cell_id[0] + self.COMPASS[dir][0]
        y1 = cell_id[1] + self.COMPASS[dir][1]

        return not self.is_open(cell_id, dir) and self.is_within_bound(x1, y1)

    def is_within_bound(self, x, y):
        # true if cell is still within bounds after the move
        return 0 <= x < self.MAZE_W and 0 <= y < self.MAZE_H

    def is_portal(self, cell):
        return tuple(cell) in self.__portals_dict

    @property
    def portals(self):
        return tuple(self.__portals)

    def get_portal(self, cell):
        if cell in self.__portals_dict:
            return self.__portals_dict[cell]
        return None

    @property
    def MAZE_W(self):
        return int(self.maze_size[0])

    @property
    def MAZE_H(self):
        return int(self.maze_size[1])

    @classmethod
    def get_walls_status(cls, cell):
        walls = {
            "U" : (cell & 0x1) >> 0,
            "R" : (cell & 0x2) >> 1,
            "D" : (cell & 0x4) >> 2,
            "L" : (cell & 0x8) >> 3,
        }
        return walls

    @classmethod
    def all_walls_intact(cls, cell):
        return cell & 0xF == 0

    @classmethod
    def num_walls_broken(cls, cell):
        walls = cls.get_walls_status(cell)
        num_broken = 0
        for wall_broken in walls.values():
            num_broken += wall_broken
        return num_broken

    @classmethod
    def __break_walls(cls, cell, dirs):
        if "U" in dirs: cell |= 0x1
        if "R" in dirs: cell |= 0x2
        if "D" in dirs: cell |= 0x4
        if "L" in dirs: cell |= 0x8
        return cell

    @classmethod
    def __get_opposite_wall(cls, dirs):

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        opposite_dirs = ""
        for dir in dirs:
            if   dir == "U": opposite_dir = "D"
            elif dir == "D": opposite_dir = "U"
            elif dir == "R": opposite_dir = "L"
            elif dir == "L": opposite_dir = "R"
            else:
                raise ValueError("The only valid directions are (U, R, D, L).")
            opposite_dirs += opposite_dir

        return opposite_dirs


class Portal:

    def __init__(self, *locations):
        self.__locations = []
        for location in locations:
            if isinstance(location, (tuple, list)):
                self.__locations.append(tuple(location))
            else:
                raise ValueError("location must be a list or a tuple.")

    def teleport(self, cell):
        if cell in self.locations:
            return self.locations[(self.locations.index(cell) + 1) % len(self.locations)]
        return cell

    def get_index(self, cell):
        return self.locations.index(cell)

    @property
    def locations(self):
        return self.__locations


class MazeEnv(object):
    """ Maze Environment for RL. """

    ACTIONS = ["U", "R", "D", "L"]
    COMPASS = {
        "U":  (0, -1),  # turn up
        "R":  (1, 0),   # turn right
        "D":  (0, 1),   # turn down
        "L":  (-1, 0)   # turn left
    }
    INIT_STATE = (0, 0)

    def __init__(self, maze_size, mode:str=None, seed:int=2021):
        
        self.maze_size = maze_size
        self.num_portal = 4 * int(mode == "plus") 
        
        self.seed_num = seed
        self.seed(self.seed_num)
        
        self._maze = Maze(maze_size=self.maze_size, 
                          has_loops=True, num_portals=self.num_portal)

        self.state = self.INIT_STATE
        self.goal = (self.maze_size[0]-1, self.maze_size[1]-1)
        self.done = False

    def seed(self, seed:int=2021):
        random.seed(seed)
        return [seed]
    
    @property
    def maze(self, ): return self._maze
    
    @property
    def legel_states(self, ):
        return [(i, j) for i in range(self.maze_size[0])
                       for j in range(self.maze_size[1])]
    
    def legal_actions(self, state):
        return self.ACTIONS
    
    def is_game_over(self, state):
        return (state[0] == self.goal[0] and state[1] == self.goal[1])
    
    def transition(self, state:tuple, action:str):
        """ Given state and action, return a list of tuples. 
            Each tuple contains 4 element, (prob, next_state, reward, done)
            where prob = P(s'|s, a).
        """
        if self.is_game_over(state):
            reward, done = 10, True
            next_state = state
        else:
            if self._maze.is_open(state, action): # will not hit a wall
                reward, done = -5/(self.maze_size[0]*self.maze_size[1]), False
                # move to next state
                next_state = (min(max(0, state[0] + self.COMPASS[action][0]), self.maze_size[0]-1), 
                            min(max(0, state[1] + self.COMPASS[action][1]), self.maze_size[1]-1))
                if self._maze.is_portal(next_state):
                    next_state = self._maze.get_portal(next_state).teleport(next_state)
            else: # hit a wall will lose more points, and not move
                reward, done = -10/(self.maze_size[0]*self.maze_size[1]), False
                next_state = state

        return [(1.0, next_state, reward, done)] # deterministically transfer to the next_state

    def reset(self,):
        self.state, self.done = self.INIT_STATE, False
        return self.state

    def step(self, action:str):
        __, next_state, reward, is_done = self.transition(self.state, action)[0]
        self.state, self.done = next_state, is_done
        return self.state, reward, is_done, {}


class MazeRLAgent(object):
    ''' A simple maze-random agent for reinforcement learning. 

    Actions:
        There are 4 discrete deterministic actions:
        - 0 / U: move up
        - 1 / R: move right
        - 2 / D: move down
        - 3 / L: move left
        When the action will cause the agent to move out of the board, 
        the agent will stay exactly where it is.

    Rewards:
        There is a default per-step reward of -5/(number of cells).
        However, if the action will lead to out-of-board or wall-hitting, 
        the reward of this step is -10/(number of cells), 
        and your agent would not move in this case.
        When the AI reaches the goal, the reward is +10.

    State:
        State space here is a square board.
        Each state is represented by an array: (col, row)
            indicates the coordinates the AI currently stands at.
    '''

    def __init__(self, gamma:float=0.99, max_episode_len:int=200, render_worker=None):
        self.gamma = gamma
        self.enable_render = (render_worker is not None)
        if self.enable_render: self.render_worker = render_worker
        self.max_episode_len = max_episode_len
        self.iteration_number = 0
    
    def render(self, env): 
        if self.enable_render: self.render_worker.render(env.state)

    def play(self, env, policy=None, strategy:str="student-force"):
        state = env.reset()
        if self.enable_render: self.render_worker.reset()

        episode_reward = 0.0
        for i in range(self.max_episode_len):
            if strategy == "random":
                action = random.choice(env.legal_actions(state))
            elif strategy == "student-force":
                assert policy is not None, "Please give a policy!"
                action = policy[state]
            elif strategy == "human":
                action_idx = input("请输入以下4个动作之一: \n\tW:向上, D:向右, S:向下, A:向左。\t输入Q退出。\n")
                action_idx = action_idx.strip()
                if action_idx in ["Q", "q", "Quit", "quit"]: return
                elif action_idx not in "WwDdSsAa": raise IndexError
                else: action = env.ACTIONS[{"W":0, "D":1, "S":2, "A":3, "w":0, "d":1, "s":2, "a":3}[action_idx]]
            else: raise NotImplementedError
            state, reward, done, _ = env.step(action)
            
            episode_reward += reward
            
            if done: break
            elif self.enable_render:
                self.render(env)
                time.sleep(0.15)
        
        return episode_reward
    
    def policy_evaluation(self, env, values, policy):
        while True:
            new_values = {s:0.0 for s in env.legel_states}
            difference = .0
            # TODO: begin your code
            for s in env.legel_states:
                prob, next_state, reward, done = env.transition(s,policy[s])[0]
                new_values[s] = reward + self.gamma * prob * values[next_state] * (1-done)
                difference += abs(new_values[s] - values[s])
            # ------ end ----------
            
            if difference < 1e-4: break
            else: 
                # TODO: begin your code
                values = new_values
                # ------ end ----------
        
        return new_values    
    
    def policy_iteration(self, env):
        """ POLICY iteration. """
        values = {s:0.0 for s in env.legel_states}
        policy = {s:random.choice(env.ACTIONS) for s in env.legel_states}

        self.iteration_number = 0
        while True:
            # TODO: begin your code
            self.iteration_number += 1
            values = self.policy_evaluation(env,values,policy)
            unchanged = True
            for s in env.legel_states:
                # 先计算最佳的Q(s,a)
                Q_s_a_max = -float('inf')
                best_a = None
                for a in env.legal_actions(s): # 对该状态的所有动作进行遍历
                    prob, next_state, reward, done = env.transition(s,a)[0] 
                    Q_s_a = reward + self.gamma * prob * values[next_state] * (1-done)
                    if Q_s_a > Q_s_a_max:
                        Q_s_a_max = Q_s_a
                        best_a = a # 最佳的动作
                # 再计算Q(s,pi[s])
                pi_s = policy[s]
                prob, next_state, reward, done = env.transition(s,pi_s)[0]
                Q_s_pi = reward + self.gamma * prob * values[next_state] * (1-done)
                # 进行比较，看是否需要更新策略
                if Q_s_a_max > Q_s_pi:
                    policy[s] = best_a
                    unchanged = False
            if unchanged:
                break
            # ------ end ----------
        
        return values, policy

    def value_iteration(self, env):
        """ VALUE iteration. """
        values = {s:0.0 for s in env.legel_states}
        policy = {s:None for s in env.legel_states}
        
        self.iteration_number = 0
        while True:
            difference = .0
            # TODO: begin your code
            self.iteration_number += 1
            import copy
            values_copy = copy.deepcopy(values) # 保存上一轮values
            for s in env.legel_states: # 遍历所有状态
                U_max = -float('inf')
                for a in env.legal_actions(s): # 对该状态的所有动作进行遍历
                    prob, next_state, reward, done = env.transition(s,a)[0] 
                    U = reward + self.gamma * prob * values_copy[next_state]*(1-done) # 值迭代，同步更新
                    if U > U_max:
                        U_max = U
                        policy[s] = a # 策略更新，在这里做掉

                values[s] = U_max # 值更新
                difference += abs(U_max - values_copy[s])
            # ------ end ----------
            
            # print(difference)
            if difference < 1e-4: 
                break
            # else: 
            #     # TODO: begin your code
            #         # 这里似乎是要做策略更新，但是我在上面把策略更新顺手做掉了
            #     # ------ end ----------

        return values, policy


if __name__ == "__main__":
    while True:
        try: seed = int(input().strip())
        except Exception: break
        env = MazeEnv(maze_size=(20, 20), mode="plus", seed=seed)
        agent = MazeRLAgent(max_episode_len=400)

        values, policy = agent.value_iteration(env)
        print("Value Iteration: {}".format(agent.iteration_number))
        for state in env.legel_states:
            print(f"{round(values[state], 4):.4f}", end=" ")
        print()

        values, policy = agent.policy_iteration(env)
        print("Policy Iteration: ")
        for state in env.legel_states:
            print(f"{round(values[state], 4):.4f}", end=" ")
        print()
