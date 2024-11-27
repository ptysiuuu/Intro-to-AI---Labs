from state import TicTacToeState
from minimax import minimax
from minimax import MinimaxParameters
import random


class Player:
    def __init__(self, is_maximizing: bool):
        self.is_maximizing = is_maximizing

    def make_move(self, state: TicTacToeState):
        if self.is_maximizing:
            params = MinimaxParameters(state, 2, float("-inf"), float("inf"), True)
        else:
            params = MinimaxParameters(state, 2, float("-inf"), float("inf"), False)
        move = minimax(params)[1] or random.choice(state.get_possible_moves())
        return move
