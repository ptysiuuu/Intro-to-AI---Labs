from state import State
from minimax import minimax
from minimax import MinimaxParameters


class Player:
    def __init__(self, is_maximizing: bool):
        self.is_maximizing = is_maximizing

    def make_move(self, state: State):
        if self.is_maximizing:
            params = MinimaxParameters(state, 2, float('-inf'), float('inf'), True)
            move = minimax(params)[1]
            return move
        params = MinimaxParameters(state, 2, float('-inf'), float('inf'), False)
        move = minimax(params)[1]
        return move
