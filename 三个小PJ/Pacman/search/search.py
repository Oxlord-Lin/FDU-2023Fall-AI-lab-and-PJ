# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"


    # 用它来存 ((position), [path])
    frontier = util.Stack()
    # 存访问过的
    explored = []

    # 如果是目标，直接返回空list
    if problem.isGoalState(problem.getStartState()):
        return []

    # 把开始state入栈
    frontier.push((problem.getStartState(), []))

    # 重复到栈为空
    while not frontier.isEmpty():
        # 拿到栈顶元素
        position, path = frontier.pop()

        # 如果访问过
        if position in explored:
            continue
        # 没访问过，入栈
        explored.append(position)

        # 如果是目标，直接返回路径
        if problem.isGoalState(position):
            return path

        # 拿到可能的下个动作
        successors = problem.getSuccessors(position)

        for nextState, action, _ in successors:

            # 如果没访问过，就计算path后入栈
            if nextState not in explored:
                new_path = path + [action]
                frontier.push((nextState, new_path))

    # 没找到， 返回空list
    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # 用它来存 ((position), [path])
    frontier = util.Queue()
    # 存访问过的
    explored = []


    # 如果是目标，直接返回空list
    if problem.isGoalState(problem.getStartState()):
        return []
    # 把开始state入队
    frontier.push((problem.getStartState(), []))

    # 重复到队列为空
    while not frontier.isEmpty():

        # 拿到队列头的元素
        position, path = frontier.pop()
        # 加入到已经访问过的
        explored.append(position)

        # 如果找到目标，返回路径
        if problem.isGoalState(position):
            return path

        # 没找到目标，拿到可能的下一步
        successors = problem.getSuccessors(position)

        for nextState, action, _ in successors:
            # 没访问过
            if nextState not in explored:
                explored.append(nextState)
                new_path = path + [action]
                frontier.push((nextState, new_path))

    # 没找到，返回空list
    return []


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    frontier.push((problem.getStartState(), []), 0)
    explored = []
    # 一直执行到frontier为空
    while not frontier.isEmpty():
        # 拿到目前最优
        curr_state, curr_path = frontier.pop()

        # 如果已经找到goal，直接返回
        if problem.isGoalState(curr_state):
            return curr_path

        # 把当前state标记为已经访问
        # explored.append(curr_state)

        # 找到可以访问的邻居
        for nextState, action, cost in problem.getSuccessors(curr_state):
            if nextState not in explored:
                explored.append(nextState)
                new_path = curr_path + [action]
                new_cost = problem.getCostOfActions(new_path)
                frontier.update((nextState, new_path), new_cost)

    # 没法到达
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    frontier = util.PriorityQueue()
    start = problem.getStartState()
    frontier.push((start, []), heuristic(start, problem))
    explored = []
    # 一直执行到frontier为空
    while not frontier.isEmpty():
        # 拿到目前最优
        curr_state, curr_path = frontier.pop()

        # 如果已经找到goal，直接返回
        if problem.isGoalState(curr_state):
            return curr_path

        # 把当前state标记为已经访问
        explored.append(curr_state)

        # 找到可以访问的邻居
        for nextState, action, cost in problem.getSuccessors(curr_state):
            if nextState not in explored:
                explored.append(nextState)
                new_path = curr_path + [action]
                new_cost = problem.getCostOfActions(new_path) + heuristic(nextState, problem)
                frontier.update((nextState, new_path), new_cost)

    # 没法到达
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
