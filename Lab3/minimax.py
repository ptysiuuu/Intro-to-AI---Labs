from dataclasses import dataclass
from state import State

@dataclass
class MinimaxParameters:
    s: State
    d: int
    alfa: float
    beta: float

def minimax(params: MinimaxParameters):
    if params.d == 0 or params.s.check_terminal():
        return params.s.evaluate_board(), params.s

    possible_moves = params.s.get_possible_moves()
    best_state = None

    if params.s.max_move:
        max_eval = float("-inf")
        player = "O"
        for move in possible_moves:
            next_state = params.s.apply_move(move, player)
            iter_params = MinimaxParameters(
                next_state,
                params.d - 1,
                params.alfa,
                params.beta,
            )
            eval, _ = minimax(iter_params)
            if eval > max_eval:
                max_eval = eval
                best_state = next_state
            params.alfa = max(params.alfa, max_eval)
            if params.alfa >= params.beta:
                break
        return max_eval, best_state
    else:
        min_eval = float("inf")
        player = "X"
        for move in possible_moves:
            next_state = params.s.apply_move(move, player)
            iter_params = MinimaxParameters(
                next_state,
                params.d - 1,
                params.alfa,
                params.beta,
            )
            eval, _ = minimax(iter_params)
            if eval < min_eval:
                min_eval = eval
                best_state = next_state
            params.beta = min(params.beta, min_eval)
            if params.alfa >= params.beta:
                break
        return min_eval, best_state
