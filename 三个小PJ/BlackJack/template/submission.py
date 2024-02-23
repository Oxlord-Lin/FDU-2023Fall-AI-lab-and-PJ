import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    # Return a value of any type capturing the start state of the MDP.
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 'Start'
        # END_YOUR_CODE

    # Return a list of strings representing actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        if state == 'Start':
            return ['Go']
        else: # 已经到终点
            return ['Stay']
        # END_YOUR_CODE

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # Remember that if |state| is an end state, you should return an empty list [].
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        if state == 'Start' and action == 'Go':
            return [('EndOne', 0.8, 0),('EndTwo', 0.2, 10)]
        else:
            return []
        # END_YOUR_CODE

    # Set the discount factor (float or integer) for your counterexample MDP.
    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return 1
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: list of integers (face values for each card included in the deck)
        multiplicity: single integer representing the number of cards with each face value
        threshold: maximum number of points (i.e. sum of card values in hand) before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look closely at this function to see an example of state representation for our Blackjack game.
    # Each state is a tuple with 3 elements:
    #   -- The first element of the tuple is the sum of the cards in the player's hand.
    #   -- If the player's last action was to peek, the second element is the index
    #      (not the face value) of the next card that will be drawn; otherwise, the
    #      second element is None.
    #   -- The third element is a tuple giving counts for each of the cards remaining
    #      in the deck, or None if the deck is empty or the game is over (e.g. when
    #      the user quits or goes bust).
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be placed into the succAndProbReward function below.
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Given a |state| and |action|, return a list of (newState, prob, reward) tuples
    # corresponding to the states reachable from |state| when taking |action|.
    # A few reminders:
    # * Indicate a terminal state (after quitting, busting, or running out of cards)
    #   by setting the deck to None.
    # * If |state| is an end state, you should return an empty list [].
    # * When the probability is 0 for a transition to a particular new state,
    #   don't include that state in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 38 lines of code, but don't worry if you deviate from this)
        totalValue, nextIndex, deckCounts = state
        result = []
        # 如果结束游戏
        if deckCounts is None: 
            return []
        # 如果选择拿牌
        if action == 'Take':
            # 如果之前偷看过，那么拿牌的结果是确定的
            if nextIndex != None:
                totalValue += self.cardValues[nextIndex]
                # 超出界限，爆掉
                if totalValue > self.threshold:
                    result.append(((0, None, None), 1, 0)) # 爆掉没奖励
                else:
                    newDeckCounts = list(deckCounts)
                    newDeckCounts[nextIndex] -= 1
                    # 如果牌拿完了
                    if sum(newDeckCounts) == 0:
                        result.append(((totalValue, None, None), 1, totalValue))
                    else:
                        result.append(((totalValue, None, tuple(newDeckCounts)), 1, 0))
            else:  #之前没偷看过
                for i in range(len(self.cardValues)):
                    newDeckCounts = list(deckCounts)
                    probability = newDeckCounts[i] / sum(deckCounts)

                    if probability == 0:
                        continue

                    newTotal = totalValue + self.cardValues[i]
                    # 爆了
                    if newTotal > self.threshold:
                        result.append(((newTotal, None, None), probability, 0))
                    else:
                        newDeckCounts[i] -= 1
                        # 牌拿完了
                        if sum(newDeckCounts) == 0:
                            result.append(((newTotal, None, None), probability, newTotal))
                        else:
                            result.append(((newTotal, None, tuple(newDeckCounts)), probability, 0))

        # 如果选择偷看
        elif action == 'Peek':
            # 不能连续偷看两次
            if nextIndex != None:
                return []
            else:
                for i in range(len(self.cardValues)):
                    # 如果有能偷看的
                    if deckCounts[i] > 0:
                        result.append(((totalValue, i, deckCounts), deckCounts[i] / sum(deckCounts), -self.peekCost))

        # 如果选择退出游戏
        elif action == 'Quit':
            result.append(((totalValue, nextIndex, None), 1, totalValue)) 
        return result
        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the
    optimal action at least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    cards = [1,3,100]
    return BlackjackMDP(cards,multiplicity=100,threshold=20,peekCost=1)

    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read utilRLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float) # defaultdict是dict的一个子类，也是字典，当出现不存在的键值时会自动添加，不会报错
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
        alpha = self.getStepSize() # 步长
        gamma = self.discount # 衰减系数
        presentValue = self.getQ(state,action) # 当前状态和动作的估计Q值
        
        if newState is not None:
            maxNewValue = -float('inf') # 找到下一个状态newState的最佳动作对应的估计Q值
            
            for newAction in self.actions(newState): # 注意，这里的actions是一个函数，详见QLearningAlgorithm的说明
                if maxNewValue < self.getQ(newState,newAction):
                    maxNewValue = self.getQ(newState,newAction)
        
        else: # newState is None，说明已经到游戏终点
            maxNewValue = 0


        residual = (reward + gamma*maxNewValue - presentValue) # 残差
        updateCoefficient = alpha * residual # 更新系数
        
        for feature, value in self.featureExtractor(state, action): 
            self.weights[feature] += updateCoefficient * value  # 对权重w的每一个分量进行更新
            # weights是defaultdict类，是一个字典，而非列表
            # value是对应于feature的值

        # END_YOUR_CODE

# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
# Small test case
smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# Large test case
largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)

def simulate_QL_over_MDP(mdp, featureExtractor):
    # NOTE: adding more code to this function is totally optional, but it will probably be useful
    # to you as you work to answer question 4b (a written question on this assignment).  We suggest
    # that you add a few lines of code here to run value iteration, simulate Q-learning on the MDP,
    # and then print some stats comparing the policies learned by these two approaches.
    # BEGIN_YOUR_CODE
    valueIteration = ValueIteration()
    valueIteration.solve(mdp)
    V_pi = valueIteration.pi # 由价值迭代所得到的最优策略
    QL = QLearningAlgorithm(mdp.actions, mdp.discount(), featureExtractor, explorationProb=0.2) # Q-learning
    util.simulate(mdp, QL, numTrials=30000, verbose=False) # 在simulate过程中会不断更新weights
    QL.explorationProb = 0 # 别忘了将Q-learning 算法的 explorationProb 设置为 0
    error, total = 0, len(mdp.states)
    for state in mdp.states:
        if V_pi[state] != QL.getAction(state):
            error += 1
    print('最优动作错误的状态数:',error,' 总的状态数:',total)
    print('错误率为： {:.3f}'.format(100 * error / total) + '%')
    # END_YOUR_CODE


############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs.
# (See identityFeatureExtractor() above for a simple example.)
# Include the following features in the list you return:
# -- Indicator for the action and the current total (1 feature).
# -- Indicator for the action and the presence/absence of each face value in the deck.
#       Example: if the deck is (3, 4, 0, 2), then your indicator on the presence of each card is (1, 1, 0, 1)
#       Note: only add this feature if the deck is not None.
# -- Indicators for the action and the number of cards remaining with each face value (len(counts) features).
#       Note: only add these features if the deck is not None.
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state

    # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
    feature = []
    feature.append(((total, action), 1))

    if counts is not None:
        feature.append(((action, tuple(int(i > 0) for i in counts)), 1))
        for k, count in enumerate(counts):
            feature.append(((action, k, count), 1))
    return feature
    # END_YOUR_CODE

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

# Original mdp
originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

# New threshold
newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

def compare_changed_MDP(original_mdp, modified_mdp, featureExtractor):
    # NOTE: as in 4b above, adding more code to this function is completely optional, but we've added
    # this partial function here to help you figure out the answer to 4d (a written question).
    # Consider adding some code here to simulate two different policies over the modified MDP
    # and compare the rewards generated by each.
    # BEGIN_YOUR_CODE
    from util import ValueIteration
    vIteration = ValueIteration()
    vIteration.solve(original_mdp)

    # 通过FixedRLAlgorithm 实例调用，基于旧的策略
    RL_old = util.FixedRLAlgorithm(vIteration.pi)
    rewards = util.simulate(original_mdp,RL_old, numTrials=10000)
    print('由旧的策略得到的平均回报为：',sum(rewards) / len(rewards))
    
    # 使用 blackjackFeatureExtractor 和默认探索概率直接在 newThresholdMDP 上模拟 Q-learning
    RL_QL = QLearningAlgorithm(original_mdp.actions, original_mdp.discount(), featureExtractor)
    rewards = util.simulate(modified_mdp,RL_QL, numTrials=10000)
    print('由基于新的特征提取器的Q-learning得到的平均回报为：',sum(rewards) / len(rewards))
    # END_YOUR_CODE

