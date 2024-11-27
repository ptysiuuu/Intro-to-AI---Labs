from state import TicTacToeState
from time import sleep
from player import Player


class Game:
    def __init__(self):
        self.game_state = TicTacToeState()
        self.max_player = Player(True)
        self.min_player = Player(False)
        self.max_next_move = True

    def display_board(self):
        for i, row in enumerate(self.game_state.board):
            print(" | ".join(row))
            if i < len(self.game_state.board) - 1:
                print("-" * (len(row) * 4 - 3))
        print()

    def get_move(self):
        if self.max_next_move:
            self.max_next_move = False
            move = self.max_player.make_move(self.game_state)
            next_state = self.game_state.apply_move(move, "O")
            self.game_state = next_state
        else:
            self.max_next_move = True
            move = self.min_player.make_move(self.game_state)
            next_state = self.game_state.apply_move(move, "X")
            self.game_state = next_state

    def play(self):
        while not self.game_state.check_terminal():
            self.get_move()
            self.display_board()
            sleep(2)
        if self.game_state.check_winner("O"):
            print("O Won!")
        elif self.game_state.check_winner("X"):
            print("X Won!")
        else:
            print("Draw!")
