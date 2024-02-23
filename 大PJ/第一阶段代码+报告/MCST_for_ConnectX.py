import math, random, time
import numpy as np
from numba import jit, njit

# random.seed(time.time())

class node():
    def __init__(self, Board, Parent, Player:int, Action = None):
        global inarow
        self.board = Board
        self.untriedLegalMoves = get_legal_move(Board) # 存储当前棋局所有可以走子的并且还没尝试过的列
        self.ChildNodes = [] # 所有已经访问过的子节点，以列表形式存储；
                                # 该节点刚被创建时，显然没有任何子节点被访问过
        self.parent = Parent  # 父节点
        self.player = Player # 本节点的下棋者是：先手or后手
        self.action = Action # 从parent到达该节点时，棋子落在哪一列
        self.win = 0 # 获胜次数，注意，该值与player有关，在backup函数中要特别小心
        self.visited = 0  # 被访问次数
        if Action:
            self.terminal = get_winner(Board, Action) is not None  # {False:这个节点还没下完，True：这个节点下完了}
            if self.terminal == True:
                self.untriedLegalMoves = []
                self.ChildNodes = []
        else:
            self.terminal = False
            

    def isTerminal(self):
        return self.terminal
    

    def fullyExpanded(self):
        """判断该节点是否被完全扩展，即是否所有legal moves所对应的子节点都被访问过"""
        return len(self.untriedLegalMoves) == 0 # 不存在未被尝试过的legal moves，就说明完全扩展

    # @jit
    def findBestChild(self,n,c=1):
        """从所有子节点中选取最优的子节点，其中n表示总的蒙特卡洛模拟次数，c是计算UCT的系数，默认为1"""
        # 这里经常出错，但还找不到原因，所以只好先用try
        try:
            bestUct = -float('inf')
            bestChild = self.ChildNodes[0]
            for child in self.ChildNodes:
                uct = child.win/child.visited + c*math.sqrt(2*math.log(n)/child.visited)
                if bestUct < uct:
                    bestChild = child
                    bestUct = uct
            return bestChild
        except IndexError:
            return self
    
    def expand(self):
        """从当前节点向向下走一步，并将子节点添加到父节点的ChildNodes属性中，返回子节点"""
        currentNode = self
        action = random.choice(currentNode.untriedLegalMoves)
        currentNode.untriedLegalMoves.remove(action) # 这个action被试过了，就从untriedLegalMoves中删去
        next_player = currentNode.player%2 + 1
        next_board = put_a_piece(currentNode.board, action, currentNode.player) 
        child = node(Board = next_board, Parent = currentNode, Player = next_player, Action=action)
        currentNode.ChildNodes.append(child)
        return child
    
    # @jit
    # def simulate(self, player:int) -> float:
    def simulate(self) -> float:
        """对当前节点进行蒙特卡洛模拟，
            返回{0：赢，1：输，0.5：平局}（相对于当前节点的落子方，negamax思想）
            """
        board = self.board
        new_player = self.player
        # 先看一下当前节点是否已经结束
        playResult = get_winner(board, self.action)
        if playResult == 0: 
            return 0.5
        if self.player == playResult: 
            return 0  # negamax
        if self.player != playResult:
            return 1 # negamax

        action = random.choice(self.untriedLegalMoves)
        while True:  
            # round = round + 1
            board = put_a_piece(board,action,new_player)
            if get_winner(board,action) in (0,1,2):
                break
            new_player = new_player%2 + 1
            action = random.choice(get_legal_move(board))
        
        # 退出循环的时候一定是到达terminal节点，也就是已经平局，或者决出胜负
        playResult = get_winner(board, action)
        if playResult == 0: # 平局各加半分
            return 0.5
        if self.player == playResult: # negamax
            return 0
        if self.player != playResult: # negamax
            return 1
        
    # @jit
    def backup(self,reward):
        """反向传播，一路往上走，更改父辈们的win和visited，使用negamax策略"""
        ## 重写的反向传播，reward是交替的，目前暂不考虑衰减系数
        node = self
        if abs(reward - 0.5) < 0.01: # 平局情况，注意浮点数的大小比较
            while node is not None: # rootNode.parent == None，此时结束循环
                node.visited += 1
                node.win += reward
                node = node.parent
        else:
            reward = round(reward)
            while node is not None:
                node.visited += 1
                node.win += reward
                node = node.parent
                reward = (reward+1)%2



class MCTree():
    def __init__(self,rootNode:node):
        self.root = rootNode

    # @jit
    def findNodeToSimulate(self,n,c=1) -> node:
        """n表示总的蒙特卡洛模拟次数，c是计算UCT时的超参数，默认为1
        该函数能够根据UCT值，返回一个【最值得进行蒙特卡洛模拟】的节点
        """
        Node = self.root # 从根节点开始寻找
        while not Node.isTerminal():
            if not Node.fullyExpanded():
                child = Node.expand()
                return child
            else:
                Node = Node.findBestChild(n,c)
        return Node

# 已完成！
@njit
def put_a_piece(board:np.array, col:int, player:int) -> np.array:
    """走子函数，board是当前棋局状态，col是棋子落下的列，player表示先手/后手
        该函数返回走子后的棋盘状态
        请先复制棋盘，后走子"""
    new_board = board.copy()
    # new_board = copy.deepcopy(board)
    # new_board = np.array(board)
    i = 0
    while board[i][col] == 0:  # 这里可能还需要处理一下
        i += 1
        if i == board.shape[0]:
            break
    new_board[i-1][col] = player
    return new_board

# 已完成！
@njit
def get_legal_move(board:np.array) -> list:
    """输入当前棋盘状态，以list形式，返回可以落子的列，默认从0开始编码"""
    # numpy的array统一从0开始编码
    col = board.shape[1] # board的列数
    legal_col = [i for i in range(col) if board[0][i] == 0]
    return legal_col  

# 已完成！
@njit
def get_winner(board:np.array, action:int) -> int:
    """输入当前棋盘状态board，需要多少个棋子相连才算胜利inarow，以及目前下棋的位置action，
    只需要考察action局部的信息即可知道是否结束，节省时间
    该函数返回胜利方，具体为：{None:还没结束，0：平局，1：先手胜利，2：后手胜利}"""
    global inarow
    
    max_col = board.shape[1]
    max_row = board.shape[0]  # 棋盘的行数和列数
    max_num = 0  # 记录行、列、对角最大的连续棋子个数
    i = 0
    while board[i][action] == 0:
        i += 1
    col = action  # 确定所下棋子的行和列(从0开始)
    row = i

    num_1 = 1  # 判断棋子所在列
    for k1 in range(row+1, max_row):  # 该棋子所在列的下方
        if board[k1][col] == board[row][col]:
            num_1 += 1
        else:
            break
    max_num = max(max_num, num_1)

    num_2 = 1  # 判断棋子所在行
    for k2 in range(col+1, max_col):  # 棋子所在行的右边
        if board[row][k2] == board[row][col]:
            num_2 += 1
        else:
            break
    for k2 in range(col-1, -1, -1):  # 棋子所在行的左边
        if board[row][k2] == board[row][col]:
            num_2 += 1
        else:
            break
    max_num = max(max_num, num_2)

    num_3 = 1  # 判断棋子所在对角线(左上右下)
    for k3 in range(1, min(max_row-row, max_col-col)):  # 棋子所在位置的右下方
        if board[row+k3][col+k3] == board[row][col]:
            num_3 += 1
        else:
            break
    for k3 in range(1, min(row+1, col+1)):  # 棋子所在位置的左上方
        if board[row-k3][col-k3] == board[row][col]:
            num_3 += 1
        else:
            break
    max_num = max(max_num, num_3)

    num_4 = 1  # 判断棋子所在对角线(左下右上)
    for k4 in range(1, min(row+1, max_col-col)):  # 棋子所在位置的右上方
        if board[row-k4][col+k4] == board[row][col]:
            num_4 += 1
        else:
            break
    for k4 in range(1, min(max_row-row, col+1)):  # 棋子所在位置的左下方
        if board[row+k4][col-k4] == board[row][col]:
            num_4 += 1
        else:
            break
    max_num = max(max_num, num_4)

    if max_num >= inarow:
        return board[row][action]
    elif row == 0 and board[0, :].all():  # 没分出输赢，且棋盘已摆满(第一行的值均不为0)
        return 0
    else:
        return None

# 已完成！
# @jit
def trans(board:list, col:int) -> np.array:
    """将一维数组转化为棋盘，board是一维数组，按照从左到右、从上到下的顺序存放，col为列数"""
    board = np.int8(board)
    new_board = board.reshape(-1, col)
    return new_board

###############################################################################################


@njit
def bestExtraTime(board:np.array,totalExtraTime):
    """输入当前的棋局状态board，以及剩余的总超额时间totalExtraTime，目前打算按照指数衰减的方式，
        返回本次落子的最合理的可用超额时间"""
    # 下面只是随便写的一个超时分配函数
    k = (np.count_nonzero(board))/(board.size)
    return min((k*totalExtraTime,5))
    # return totalExtraTime/2

# @jit
def bestOpponent(board:np.array,opponent):
    """判断是否存在一个位置，当对手在此位置落子，则对手立刻获胜"""
    for i in get_legal_move(board):
        # print(i)
        playResult = get_winner(put_a_piece(board,i,opponent),i)
        if playResult == opponent:
            return i
    # print('get out of bestOpponent')
    return -10

# @njit
def MCST_agent(obs,config):
    startTime = time.time() # 开始时间
    player = obs.mark
    col = config.columns
    # row = config.rows
    global inarow
    inarow = config.inarow
    board = trans(obs.board,col) # 将一维棋盘转换为二维棋盘
    stepTime = config.actTimeout # 本次默认思考时间，由config给定，似乎无法自行更改
    nonzero_count = np.count_nonzero(board)

    # 第一次落子的随机性太大，搜索的意义不大，所以不分配超额时间
    if nonzero_count <= 1:
        return board.shape[1]//2
    else:
        totalExtraTime = obs.remainingOverageTime
        extraTime = bestExtraTime(board,totalExtraTime) # 本次可用的超额时间
        timeOut = stepTime + extraTime# 本次思考时间
        timeOut = 2

    # 检查是否只有一列可下，若是，则无需模拟直接返回action
    if len(get_legal_move(board)) == 1:
        return get_legal_move(board)[0]
    
    # 先检查是否有马上能赢的落子方法（启发式）
    for action in get_legal_move(board):
        if player == get_winner(put_a_piece(board,action,player),action):
            return action
    
    # 再进行一次“邻近威胁”的检查，看一下对手是不是马上就能连成线
    # 如果是，就要立即阻止对手！
    opponent = player%2 + 1
    action = bestOpponent(board,opponent)
    # action = 0
    if action >= 0:
        return action

    rootNode = node(Board=board,Parent=None, Player=player) # rootNode的特征就是，没有父母
    mcst = MCTree(rootNode) # 建立Monte Carlo树，并且插入根节点
    n = 0  # 总的蒙特卡洛模拟次数
    # global c
    c = 1 # 计算UCT时的系数（超参数）
    k = (np.count_nonzero(board))/(board.size)
    c = 1 + k*0.1
    while time.time() - startTime < timeOut: # 还有剩余时间
        n = n + 1 # 模拟次数加一
        node_to_simulate = mcst.findNodeToSimulate(n,c)
        reward = node_to_simulate.simulate()
        node_to_simulate.backup(reward)
    
    try:   # 防止因为意外的出错而输掉比赛，无论如何要返回一个action
        best = mcst.root.findBestChild(n,c=0)
        return best.action
    except Exception:
        return random.choice(get_legal_move(board)) 

    