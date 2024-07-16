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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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

        # Initialize score from the successor's game state score
        tot_score = successorGameState.getScore()


        # Consider food distance
        foodList = newFood.asList()
        if foodList:
            minFoodDistance = min(manhattanDistance(newPos, foodPos) for foodPos in foodList)
            tot_score += 1.0 / minFoodDistance

        # Consider ghost positions
        for ghostState in newGhostStates:
            ghostPos = ghostState.getPosition()
            if ghostState.scaredTimer == 0:  # Ghost is not scared
                ghostDistance = manhattanDistance(newPos, ghostPos)
                if ghostDistance > 0:
                    tot_score -= 1.0 / ghostDistance

            else:  # Ghost is scared
                tot_score += 1.0 / manhattanDistance(newPos, ghostPos)

        return tot_score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        def minimax(agentIndex, depth, gameState):
            # Check for terminal state
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            
            

            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            

            else:
                return minValue(agentIndex, depth, gameState)

        def getNextAgent(agentIndex):
            # Get the next agent's index, looping back to Pacman after the last ghost
            turn = (agentIndex + 1) % gameState.getNumAgents()
            return turn 

        def getNextDepth(agentIndex, depth):
            # Increment depth only after all agents (including ghosts) have moved
            return depth + 1 if getNextAgent(agentIndex) == 0 else depth

        def maxValue(agentIndex, depth, gameState):
            v = float('-inf')
            for action in gameState.getLegalActions(agentIndex):

                successorState = gameState.generateSuccessor(agentIndex, action)
                v = max(v, minimax(getNextAgent(agentIndex), getNextDepth(agentIndex, depth), successorState))
            return v

        def minValue(agentIndex, depth, gameState):
            val = float('inf')
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                
                val = min(val, minimax(getNextAgent(agentIndex), getNextDepth(agentIndex, depth), successorState))
            
            return val

        # Initialize minimax for Pacman
        bestAction = None
        bestValue = float('-inf')
        for action in gameState.getLegalActions(0):
            successorState = gameState.generateSuccessor(0, action)
            value = minimax(1, 0, successorState)
            if value > bestValue:
                bestValue = value
                bestAction = action
        
        return bestAction
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def minimax_alpha_beta(agentIndex, depth, state, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn (maximizing player)
                return max_value(agentIndex, depth, state, alpha, beta)
            else:  # Ghosts' turns (minimizing players)
                return min_value(agentIndex, depth, state, alpha, beta)

        def next_agent(agentIndex):
            return (agentIndex + 1) % gameState.getNumAgents()

        def next_depth(agentIndex, depth):
            return depth + 1 if next_agent(agentIndex) == 0 else depth

        def max_value(agentIndex, depth, state, alpha, beta):
            val = float('-inf')
            legal_actions = state.getLegalActions(agentIndex)
            for action in legal_actions:
                successor_state = state.generateSuccessor(agentIndex, action)
                val = max(val, minimax_alpha_beta(next_agent(agentIndex), next_depth(agentIndex, depth), successor_state, alpha, beta))
                if val > beta:  # Prune if v surpasses beta
                    return val
                alpha = max(alpha, val)
            return val

        def min_value(agentIndex, depth, state, alpha, beta):
            v = float('inf')
            legal_actions = state.getLegalActions(agentIndex)
            for action in legal_actions:
                successor_state = state.generateSuccessor(agentIndex, action)
                v = min(v, minimax_alpha_beta(next_agent(agentIndex), next_depth(agentIndex, depth), successor_state, alpha, beta))
                if v < alpha:  # Prune if v drops below alpha
                    return v
                beta = min(beta, v)
            return v

        # Initialize alpha and beta for the root node
        alpha = float('-inf')
        beta = float('inf')
        best_actio = None
        best_value = float('-inf')
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(0, action)
            value = minimax_alpha_beta(1, 0, successor_state, alpha, beta)
            if value > best_value:
                best_value = value

                best_actio = action
            alpha = max(alpha, best_value)  # Update alpha after evaluating each action

        return best_actio



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """

        def expectimax(agentIndex, depth, state):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agentIndex == 0:  # Pacman's turn maximizing player
                return max_value(agentIndex, depth, state)
            else:  # Ghosts' turns expectation over min values
                return exp_value(agentIndex, depth, state)

        def next_agent(agentIndex):
            return (agentIndex + 1) % gameState.getNumAgents()

        def next_depth(agentIndex, depth):
            if next_agent(agentIndex) == 0:
                return depth +  1
            else: 
                return depth


        def max_value(agentIndex, depth, state):
            val = float('-inf')

            legal_actions = state.getLegalActions(agentIndex)
            for action in legal_actions:
                successor_state = state.generateSuccessor(agentIndex, action)
                val = max(val, expectimax(next_agent(agentIndex), next_depth(agentIndex, depth), successor_state))
            return val

        def exp_value(agentIndex, depth, state):
            v = 0
            legal_actions = state.getLegalActions(agentIndex)
            num_actions = len(legal_actions)
            for action in legal_actions:
                successor_state = state.generateSuccessor(agentIndex, action)
                v += expectimax(next_agent(agentIndex), next_depth(agentIndex, depth), successor_state) / num_actions
            return v

        # Initialize best action and value for Pacman
        best_action = None
        best_value = float('-inf')
        legal_actions = gameState.getLegalActions(0)
        for action in legal_actions:
            successor_state = gameState.generateSuccessor(0, action)
            value = expectimax(1, 0, successor_state)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function.
    """

    # Gamestate info
    pacmanPosition = currentGameState.getPacmanPosition()

    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    # Start with the current game score
    pac_score = currentGameState.getScore()


    # Evaluate distances to food
    foodDistances = [manhattanDistance(pacmanPosition, food) for food in foodGrid.asList()]
    if foodDistances:
        closestFoodDistance = min(foodDistances)
        pac_score += 1.0 / closestFoodDistance

    # Evaluate distances to ghosts
    for ghostState, scaredTime in zip(ghostStates, scaredTimes):
        ghostPos = ghostState.getPosition()
        distance_To_Ghost = manhattanDistance(pacmanPosition, ghostPos)

        if scaredTime > 0:
            # If the ghost is scared, prioritize eating it
            pac_score += 1.0 / (distance_To_Ghost + 1)  # Add 1 to avoid division by zero
        else:
            # If the ghost is not scared, prioritize avoiding it
            pac_score -= 1.0 / (distance_To_Ghost + 1)  # Add 1 to avoid division by zero

    return pac_score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
