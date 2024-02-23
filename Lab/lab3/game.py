# -*- encoding: utf-8 -*-
# Author: Jiwen Zhang
# Time: 04/08 23:33
# Version: 1.0
# Contact: jwzhang16@fudan.edu.cn

import pygame
import numpy as np


class MazeRender(object):

    def __init__(self, 
        maze,
        screen_name:str="Artificial Intelligence - Lab3: Maze", 
        screen_size:tuple=(640, 640),
    ):

        pygame.init()
        pygame.display.set_caption(screen_name)
        self.clock = pygame.time.Clock()
        # by default, the robot stays at the left-up corner
        self._robot = np.array((0, 0))
        self.maze = maze
        self.maze_size = self.maze.maze_size

        self.screen = pygame.display.set_mode(screen_size)
        self.screen_size = (screen_size[0]-1, screen_size[1]-1)

        self.init_screen()

    def init_screen(self, ):
        # Create a background
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((255, 255, 255))

        # Create a layer for the maze
        self.maze_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.maze_layer.fill((0, 0, 0, 0,))

        self.__draw_maze()                      # show the maze
        self.__draw_portals()                   # show the portals
        self.__draw_entrance()                  # show the entrance
        self.__draw_goal()                      # show the goal
        self.__draw_robot(transparency=125)     # show the robot
        
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.maze_layer,(0, 0))
        pygame.display.flip()
    
    def __draw_maze(self):
        
        line_colour = (0, 0, 0, 255)

        # drawing the horizontal lines
        for y in range(self.maze.MAZE_H + 1):
            pygame.draw.line(self.maze_layer, line_colour, (0, y * self.CELL_H),
                             (self.SCREEN_W, y * self.CELL_H))

        # drawing the vertical lines
        for x in range(self.maze.MAZE_W + 1):
            pygame.draw.line(self.maze_layer, line_colour, (x * self.CELL_W, 0),
                             (x * self.CELL_W, self.SCREEN_H))

        # breaking the walls
        for x in range(len(self.maze.maze_cells)):
            for y in range(len(self.maze.maze_cells[x])):
                # check the which walls are open in each cell
                walls_status = self.maze.get_walls_status(self.maze.maze_cells[x][y])
                dirs = ""
                for dir, open in walls_status.items():
                    if open:
                        dirs += dir
                self.__cover_walls(x, y, dirs)
    
    def __cover_walls(self, x, y, dirs, colour=(0, 0, 255, 15)):
        
        dx = x * self.CELL_W
        dy = y * self.CELL_H

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        for dir in dirs:
            if dir == "D":
                line_head = (dx + 1, dy + self.CELL_H)
                line_tail = (dx + self.CELL_W - 1, dy + self.CELL_H)
            elif dir == "U":
                line_head = (dx + 1, dy)
                line_tail = (dx + self.CELL_W - 1, dy)
            elif dir == "L":
                line_head = (dx, dy + 1)
                line_tail = (dx, dy + self.CELL_H - 1)
            elif dir == "R":
                line_head = (dx + self.CELL_W, dy + 1)
                line_tail = (dx + self.CELL_W, dy + self.CELL_H - 1)
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            pygame.draw.line(self.maze_layer, colour, line_head, line_tail)

    def __draw_portals(self, transparency=160):

        colour_range = np.linspace(0, 255, len(self.maze.portals), dtype=int)
        colour_i = 0
        for portal in self.maze.portals:
            colour = ((100 - colour_range[colour_i])% 255, colour_range[colour_i], 0)
            colour_i += 1
            for location in portal.locations:
                self.__colour_cell(location, colour=colour, transparency=transparency)

    def __colour_cell(self, cell, colour, transparency):

        if not (isinstance(cell, (list, tuple, np.ndarray)) and len(cell) == 2):
            raise TypeError("cell must a be a tuple, list, or numpy array of size 2")

        x = int(cell[0] * self.CELL_W + 0.5 + 1)
        y = int(cell[1] * self.CELL_H + 0.5 + 1)
        w = int(self.CELL_W + 0.5 - 1)
        h = int(self.CELL_H + 0.5 - 1)
        pygame.draw.rect(self.maze_layer, colour + (transparency,), (x, y, w, h))
    
    def __draw_robot(self, colour=(238,180,180), transparency=255):
        
        x = int(self._robot[0] * self.CELL_W + self.CELL_W * 0.5 + 0.5)
        y = int(self._robot[1] * self.CELL_H + self.CELL_H * 0.5 + 0.5)
        r = int(min(self.CELL_W, self.CELL_H)/5 + 0.5)

        pygame.draw.circle(self.maze_layer, colour + (transparency,), (x, y), r)

    def __draw_entrance(self, colour=(0, 0, 150), transparency=235):

        self.__colour_cell(self.entrance, colour=colour, transparency=transparency)

    def __draw_goal(self, colour=(150, 0, 0), transparency=235):

        self.__colour_cell(self.goal, colour=colour, transparency=transparency)

    def render(self, state:tuple):
        # update the drawing
        self.__draw_robot(transparency=0)
        # move the robot
        self._robot = np.array(state)
        self.__draw_robot(transparency=255)

        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.maze_layer,(0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def reset(self,):
        self.__draw_robot(transparency=0)
        self._robot = np.zeros(2, dtype=int)
        self.__draw_robot(transparency=255)

    def quit(self,):
        pygame.display.quit()
        pygame.quit()

    @property
    def entrance(self):
        return np.array((0, 0))
    
    @property
    def goal(self):
        return np.array(self.maze_size) - np.array((1, 1))

    @property
    def robot(self):
        return self._robot

    @property
    def SCREEN_SIZE(self):
        return self.screen_size

    @property
    def SCREEN_W(self):
        return int(self.SCREEN_SIZE[0])

    @property
    def SCREEN_H(self):
        return int(self.SCREEN_SIZE[1])

    @property
    def CELL_W(self):
        return float(self.SCREEN_W) / float(self.maze.MAZE_W)

    @property
    def CELL_H(self):
        return float(self.SCREEN_H) / float(self.maze.MAZE_H)


if __name__ == "__main__":
    from maze_template import MazeRLAgent, MazeEnv

    env = MazeEnv(maze_size=(20, 20), mode="plus", seed=2001)
    render_worker = MazeRender(env.maze)
    agent = MazeRLAgent(max_episode_len=400, render_worker=render_worker)

    play_mode = input("请选择游戏模式：human / study\n").strip()
    if play_mode == "human":
        agent.play(env, strategy="human")
    elif play_mode == "study":
        values, policy = agent.policy_iteration(env)
        agent.play(env, policy, strategy="student-force")
    else: raise NotImplementedError("模式选择错误！")
    render_worker.quit()

    pass
