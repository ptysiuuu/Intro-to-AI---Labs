from state import State
from minimax import minimax, MinimaxParameters
from time import sleep
import random


class Game:
    def __init__(self):
        self.game_state = State()

    def display_board(self):
        for i, row in enumerate(self.game_state.board):
            print(" | ".join(row))
            if i < len(self.game_state.board) - 1:
                print("-" * (len(row) * 4 - 3))
        print()

    def make_move(self):
        params = MinimaxParameters(self.game_state, 4, float("-inf"), float("inf"))
        new_state = minimax(params)[1]
        if new_state is None:
            player = 'O' if self.game_state.max_move else 'X'
            move = random.choice(self.game_state.get_possible_moves())
            new_state = self.game_state.apply_move(move, player)
        self.game_state = new_state

    def play(self):
        while not self.game_state.check_terminal():
            self.make_move()
            self.display_board()
            sleep(2)
        if self.game_state.evaluate_board() == 1:
            print('O Won!')
        elif self.game_state.evaluate_board() == -1:
            print('X Won!')
        else:
            print('Draw!')
