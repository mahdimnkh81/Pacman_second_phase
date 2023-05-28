# multiAgents.py
# --------------
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
import sys

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newCapsule = successorGameState.getCapsules()
        foodScore, minDistance = foodScoreEstimate(newFood, newPos)
        capsuleScore = capsuleScoreEstimate(newCapsule, minDistance, newPos)
        score = successorGameState.getScore()
        postion = positionEstimate(currentGameState.getPacmanPosition(), newPos=newPos)
        ghostScore = ghostStateEstimate(newGhostStates, newPos, minDistance)
        scaredScore = scaredScoreEstimate(newGhostStates, minDistance, newPos)
        scaredTimer = sum(newScaredTimes)
        return postion + (ghostScore * 1.42) + (scaredTimer * 1.32) + (score * 1.10) + (foodScore * 1.45 ) + (scaredScore * 1.85) + (capsuleScore * 0.4)

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        import sys
        num_agents = gameState.getNumAgents()
        def max_value(index, state, depth):
            if state.isWin() or state.isLose() or depth + 1 == self.depth:
                return self.evaluationFunction(state)
            v = -sys.maxsize
            legalActions = state.getLegalActions(index)
            ai = (index + 1) % num_agents
            for action in legalActions:
                successor = state.generateSuccessor(index, action)
                v = max(v, min_value(ai, successor, depth + 1))
            return v
        def min_value(index, state, depth):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = sys.maxsize
            legalActions = state.getLegalActions(index)
            ai = (index + 1) % num_agents
            if ai == 0:
                for action in legalActions:
                    successor = state.generateSuccessor(index, action)
                    v = min(v, max_value(ai, successor, depth))
            else:
                for action in legalActions:
                    successor = state.generateSuccessor(index, action)
                    v = min(v, min_value(ai, successor, depth))
            return v

        legalActions = gameState.getLegalActions(0)
        act = max(legalActions, key=lambda action: min_value(1, gameState.generateSuccessor(0, action), 0))
        return act
        # util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        import sys
        num_agents = gameState.getNumAgents()

        def max_value(index, state, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth+1 == self.depth:
                return self.evaluationFunction(state)
            v = -sys.maxsize
            legalActions = state.getLegalActions(index)
            a = alpha
            ai = (index + 1) % num_agents  # agent index
            for action in legalActions:
                successor = state.generateSuccessor(index, action)
                v = max(v, min_value(ai, successor, depth+1, a, beta))
                if v > beta:
                    return v
                a = max(a, v)
            return v

        def min_value(index, state, depth, alpha, beta):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v = sys.maxsize
            legalActions = state.getLegalActions(index)
            b = beta
            ai = (index + 1) % num_agents  # agent index
            for action in legalActions:
                successor = state.generateSuccessor(index, action)
                if ai == 0:
                    v = min(v, max_value(ai, successor, depth, alpha, b))
                    if v < alpha:
                        return v
                    b = min(b, v)
                else:
                    v = min(v, min_value(ai, successor, depth, alpha, b))
                    if v < alpha:
                        return v
                    b = min(b, v)
            return v

        alpha = -sys.maxsize
        beta = sys.maxsize
        score = -sys.maxsize
        act = None
        legalActions = gameState.getLegalActions(0)
        for action in legalActions:
            state = gameState.generateSuccessor(0, action)
            if min_value(1, state, 0, alpha, beta) > score:
                act = action
                score = min_value(1, state, 0, alpha, beta)
            if score > beta:
                return act
            alpha = max(alpha, score)
        return act
        # util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        agentsNum = gameState.getNumAgents()

        def isTerminalState(gameState, currentDepth, agentIndex):
            return currentDepth == self.depth or \
                   not gameState.getLegalActions(agentIndex)

        def maxValue(gameState, currentDepth, agentIndex):
            bestResult = [-999999]

            if isTerminalState(gameState, currentDepth, agentIndex):
                bestResult[0] = self.evaluationFunction(gameState)
                return bestResult

            for a in gameState.getLegalActions(agentIndex=agentIndex):
                v = bestResult[0]
                nextResult = expValue(gameState.generateSuccessor(agentIndex=agentIndex, action=a), currentDepth,
                                      agentIndex=1)
                next_v = nextResult[0]
                if v < next_v:
                    bestResult[0] = next_v
                    try:
                        bestResult[1] = a
                    except IndexError:  # for the first iteration, we haven't pushed the action before
                        bestResult.append(a)

            return bestResult

        def expValue(gameState, currentDepth, agentIndex):
            bestResult = [0]

            if isTerminalState(gameState, currentDepth, agentIndex):
                bestResult[0] = self.evaluationFunction(gameState)
                return bestResult

            if agentIndex < agentsNum - 1:  # then we should determine other ghosts' decision
                for a in gameState.getLegalActions(agentIndex=agentIndex):
                    p = 1. / len(gameState.getLegalActions(agentIndex=agentIndex))
                    v = expValue(gameState.generateSuccessor(agentIndex=agentIndex, action=a), currentDepth,
                                 agentIndex=agentIndex + 1)
                    bestResult[0] += p * v[0]

            else:  # the next agent is pacman
                for a in gameState.getLegalActions(agentIndex=agentIndex):
                    p = 1. / len(gameState.getLegalActions(agentIndex=agentIndex))
                    v = maxValue(gameState.generateSuccessor(agentIndex=agentIndex, action=a), currentDepth + 1,
                                 agentIndex=0)
                    bestResult[0] += p * v[0]

            return bestResult

        return maxValue(gameState, currentDepth=0, agentIndex=0)[1]  # second element is the best action
        # util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()
    
    "*** YOUR CODE HERE ***"
    from util import manhattanDistance as mDist
    newPos = currentGameState.getPacmanPosition()
    newFoodPos = (currentGameState.getFood()).asList()
    newGhostStates = currentGameState.getGhostStates()
    newGhostPos = [ghostState.getPosition() for ghostState in newGhostStates]
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    evaluatedValue = 0
    for foodDist in newFoodPos:
        dist = mDist(newPos, foodDist)
        if dist < 5:
            evaluatedValue += 6 - dist
        else:
            evaluatedValue += 1

    for ghostDist in newGhostPos:
        dist = mDist(newPos, ghostDist)
        if dist == 0:
            evaluatedValue = -1 * evaluatedValue
        elif dist < 5:
            evaluatedValue -= (6 - dist) * 1.5
        else:
            evaluatedValue += 1

    return currentGameState.getScore() + evaluatedValue + sum(newScaredTimes)

# Abbreviation
better = betterEvaluationFunction
def foodScoreEstimate(newFood, NewPositions):
    foodDistance = []
    for i in newFood.asList():
      if newFood[i[0]][i[1]]:
        foodDistance.append(manhattanDistance(NewPositions, i))
    foodScore = 0
    minDistance = 0
    if foodDistance:
        minDistance = min(foodDistance)
        if minDistance == 0:
            foodScore = 100
        else:
            foodScore = 10/minDistance
    return foodScore, minDistance

def ghostStateEstimate(newGhostStates, newPos, minDistance):
    activeGhostPositions = [i.getPosition() for i in newGhostStates if i.scaredTimer == 0]
    ghostScore = 100
    maxGhostDistance = -1
    minGhostDistance = -1
    if newPos in activeGhostPositions:
        ghostScore = -1000
    activeGhostDistance = [manhattanDistance(newPos, i) for i in activeGhostPositions]

    if activeGhostDistance:
        maxGhostDistance = max(activeGhostDistance)
        minGhostDistance = min(activeGhostDistance)
        if minGhostDistance <= 1:
            ghostScore = -1000
            minGhostDistance = -1
        else:
            if minGhostDistance < minDistance:
                ghostScore = -100
            ghostScore = maxGhostDistance
    return ghostScore

def capsuleScoreEstimate(newCapsule, minDistance, newPos):
    capsuleScore = 0
    if len(newCapsule) >= 1:
        capsuleDistance = min([manhattanDistance(newPos, i) for i in newCapsule])
    if newPos in newCapsule and capsuleDistance < minDistance:
        capsuleScore = 2000
    return capsuleScore

def scaredScoreEstimate(newGhostStates, minDistance, newPos):
    scaredGhostPositions = [i.getPosition() for i in newGhostStates if i.scaredTimer != 0]
    scaredScore = 0
    if len(scaredGhostPositions) > 0:
        if newPos in scaredGhostPositions:
            scaredScore = 20000
        else:
            scaredGhostDistance = min([manhattanDistance(newPos, i) for i in scaredGhostPositions])
            if scaredGhostDistance < minDistance:
                scaredScore = minDistance - scaredGhostDistance
    return scaredScore

def positionEstimate(currentPos, newPos):
    if currentPos == newPos:
        return -100
    else:
        return 0

