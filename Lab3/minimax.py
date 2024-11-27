from dataclasses import dataclass
from abc import abstractmethod, ABC


class MinimaxState(ABC):
    @abstractmethod
    def evaluate(self):
        '''
        This method should return a evaluation of the current state.
        '''
        pass

    @abstractmethod
    def apply_move(self, move, player):
        '''
        This method should return a new MinimaxState object,
        that represents the state after a move by a player is made
        where move and player are given as arguments.
        '''
        pass

    @abstractmethod
    def get_possible_moves(self):
        '''
        This method should return all possible moves that
        can be applied to the current state.
        '''
        pass

    @abstractmethod
    def check_terminal(self):
        '''
        This method should return True if the current state is terminal.
        '''
        pass


@dataclass
class MinimaxParameters:
    s: MinimaxState
    d: int
    alfa: float
    beta: float
    is_max_move: bool

def minimax(params: MinimaxParameters):
    if params.d == 0 or params.s.check_terminal():
        return params.s.evaluate(), None

    possible_moves = params.s.get_possible_moves()
    best_move = None

    if params.is_max_move:
        max_eval = float("-inf")
        player = "O"
        for move in possible_moves:
            iter_params = MinimaxParameters(
                params.s.apply_move(move, player),
                params.d - 1,
                params.alfa,
                params.beta,
                not params.is_max_move
            )
            eval, _ = minimax(iter_params)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            params.alfa = max(params.alfa, max_eval)
            if params.alfa >= params.beta:
                break
        return max_eval, best_move
    else:
        min_eval = float("inf")
        player = "X"
        for move in possible_moves:
            iter_params = MinimaxParameters(
                params.s.apply_move(move, player),
                params.d - 1,
                params.alfa,
                params.beta,
                not params.is_max_move
            )
            eval, _ = minimax(iter_params)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            params.beta = min(params.beta, min_eval)
            if params.alfa >= params.beta:
                break
        return min_eval, best_move
