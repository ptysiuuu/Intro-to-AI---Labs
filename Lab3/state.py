from typing import Literal
from copy import deepcopy
from minimax import MinimaxState


class IncorrectMove(Exception):
    pass


class TicTacToeState(MinimaxState):
    def __init__(self, board=[[" ", " ", " "], [" ", " ", " "], [" ", " ", " "]]):
        self.board = board

    def apply_move(self, cords: tuple[int, int], player: Literal["X", "O"]):
        n = cords[0]
        p = cords[1]
        try:
            if self.board[n][p] != " ":
                raise IncorrectMove("This move is impossible")
            else:
                new_board = deepcopy(self.board)
                new_board[n][p] = player
                return TicTacToeState(new_board)
        except IndexError:
            raise IncorrectMove("This move is impossible")

    def get_possible_moves(self):
        all_moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == " ":
                    all_moves.append((i, j))
        return all_moves

    def check_winner(self, symbol: Literal["X", "O"]):
        for row in self.board:
            if row == [symbol, symbol, symbol]:
                return True
        for col in range(3):
            if all(self.board[row][col] == symbol for row in range(3)):
                return True
        if all(self.board[i][i] == symbol for i in range(3)) or all(
            self.board[i][2 - i] == symbol for i in range(3)
        ):
            return True
        return False

    def check_terminal(self):
        if self.check_winner("X") or self.check_winner("O"):
            return True

        if all(cell != " " for row in self.board for cell in row):
            return True

        return False

    def evaluate(self):
        if self.check_winner("O"):
            return 100
        elif self.check_winner("X"):
            return -100
        heuristic_values = [[3, 2, 3], [2, 4, 2], [3, 2, 3]]
        total_sum = 0
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == "O":
                    total_sum += heuristic_values[i][j]
                elif self.board[i][j] == "X":
                    total_sum -= heuristic_values[i][j]
        return total_sum
