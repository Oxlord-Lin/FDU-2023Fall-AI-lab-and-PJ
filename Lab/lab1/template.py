import heapq
import sys

class PriorityQueue:

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            assert type(i) == node, 'i must be node'
            if i.state == item.state:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)


class node:
    """define node"""

    def __init__(self, state, parent, path_cost, action,):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost
        self.action = action


class problem:
    """searching problem"""

    def __init__(self, initial_state, actions):
        self.initial_state = initial_state
        self.actions = actions

    def search_actions(self, state):
        """Search actions for the given state.
        Args:
            state: a string e.g. 'A'

        Returns:
            a list of action string list
            e.g. [['A', 'B', '2'], ['A', 'C', '3']]
        """
        ################################# Your code here ###########################
        All_actions = self.actions # 所有的action
        search_actions_list = []
        for act in All_actions:
            if act[0] == state:
                search_actions_list.append(act)
        return search_actions_list
        # raise Exception	

    def solution(self, node):  # 应该是最后呈现结果时使用的？
        """Find the path & the cost from the beginning to the given node.

        Args:
            node: the node class defined above.

        Returns:
            ['Start', 'A', 'B', ....], Cost
        """
        ################################# Your code here ###########################
        path = [node.state]
        parent_node = node.parent
        while parent_node != '':
            path.append(parent_node.state)
            parent_node = parent_node.parent
        path.reverse()
        return path, node.path_cost
        # raise Exception	

    def transition(self, state, action):
        """Find the next state from the state adopting the given action.

        Args:
            state: 'A'
            action: ['A', 'B', '2']

        Returns:
            string, representing the next state, e.g. 'B'
        """
        ################################# Your code here ###########################
        if state == action[0]:
            return action[1]
        raise Exception

    def goal_test(self, state):
        """Test if the state is goal

        Args:
            state: string, e.g. 'Goal' or 'A'

        Returns:
            a bool (True or False)
        """

        ################################# Your code here ###########################
        return state == 'Goal'
        # raise Exception	

    def step_cost(self, state1, action, state2):
        if (state1 == action[0]) and (state2 == action[1]):
            return int(action[2])
        else:
            print("Step error!")
            sys.exit()

    def child_node(self, node_begin, action):
        """Find the child node from the node adopting the given action

        Args:
            node_begin: the node class defined above. 
            action: ['A', 'B', '2']

        Returns:
            a node as defined above  [child's name? string?]
        """
        ################################# Your code here ###########################
        if node_begin.state == action[0]:
            return action[1]
        raise Exception


def UCS(problem):
    """Using Uniform Cost Search to find a solution for the problem.

    Args:
        problem: problem class defined above.

    Returns:
        a list of strings representing the path, along with the path cost as an integer.
            e.g. ['A', 'B', '2'], 5
        if the path does not exist, return 'Unreachable'
    """
    node_test = node(problem.initial_state, '', 0, '') 
    frontier = PriorityQueue()  # 从中选取最小的进行探索
    frontier.push(node_test, node_test.path_cost)
    state2node = {node_test.state: node_test} 
    # 字典里存储着所有的在explored里的节点，以及在frontier的节点（这些在frontier里的节点的path_cost有可能会被更新）
    explored = [] 

    ################################# Your code here ###########################
    while True:
        if frontier.isEmpty(): # 如果已经没有可以被发现的节点，则说明没找到
            return 'Unreachable', 0
        node_test = frontier.pop() # 将代价最低的节点弹出
        # state2node[node_test.state] = node_test
        if problem.goal_test(node_test.state): 
            # 注意，本算法在弹出的时才对其是否为goal进行确认；因此要在“同层”的都进入优先队列后才能找到goal；
            # 当然，也可以在入队之前就进行检查
            return problem.solution(node_test)
        explored.append(node_test.state) # 记录这个被探索过的节点
        for act in problem.search_actions(node_test.state):
            child_name = problem.child_node(node_test, act) # 找到所有邻居节点的名称
            temp_path_cost = node_test.path_cost + problem.step_cost(node_test.state,act,child_name)
            if child_name in explored:
                continue # 已经被探索过，凡是在explred里的节点的path_cost都不会再改变
            elif child_name in state2node.keys(): 
                # 如果不在explored但在state2.node.keys()中，则说明在frontier中出现过
                # 该节点可能要更新path_cost与parent
                if state2node[child_name].path_cost > temp_path_cost:
                    state2node[child_name].path_cost = temp_path_cost
                    state2node[child_name].parent = node_test
                    frontier.update(state2node[child_name],temp_path_cost)  
                    # frontier存储的是被扩展但尚未探索的节点；但这些节点的path_cost与parent需要手动更新
                    # 更新的方式为从字典里往frontier里update这个节点的更短的path_cost与新的parent
            else: # 第一次被发现
                child = node(child_name,node_test,temp_path_cost,'')
                state2node[child_name] = child
                frontier.update(child,temp_path_cost)


if __name__ == '__main__':
    Actions = []
    while True:
        a = input().strip()
        if a != 'END':
            a = a.split()
            Actions += [a]
        else:
            break
    graph_problem = problem('Start', Actions)
    answer, path_cost = UCS(graph_problem)
    s = "->"
    if answer == 'Unreachable':
        print(answer)
    else:
        path = s.join(answer)
        print(path)
        print(path_cost)
