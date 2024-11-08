from state import State


def test_state_get_possible_moves():
    state = State()
    assert len(state.get_possible_moves()) == 9


def test_state_get_all_possible_moves():
    state = State()
    new_state = state.apply_move((0, 0), "X")
    assert len(new_state.get_possible_moves()) == 8


def test_terminal():
    state = State()
    state.board = [["X", "X", "O"], ["O", "O", "X"], ["X", "O", "O"]]
    assert state.check_terminal()


def test_winner_horizontal():
    state = State()
    state.board = [["X", "X", "X"], ["O", "O", "X"], ["X", "O", "O"]]
    assert state.check_winner("X")


def test_winner_vertical():
    state = State()
    state.board = [["X", "O", "X"], ["O", "O", "X"], ["X", "O", "O"]]
    assert state.check_winner("O")


def test_winner_diagonal():
    state = State()
    state.board = [["X", "O", "X"], ["O", "X", "O"], ["O", "O", "X"]]
    assert state.check_winner("X")


def test_apply_move():
    state = State()
    new_state = state.apply_move((0, 0), "X")
    assert new_state.board[0][0] == "X"


def test_evaluate_board_min():
    state = State()
    state.board = [["X", "O", "X"], ["O", "X", "O"], ["O", "O", "X"]]
    assert state.evaluate_board() == -1


def test_evaluate_board_max():
    state = State()
    state.board = [["X", "O", "X"], ["O", "O", "O"], ["O", "X", "X"]]
    assert state.evaluate_board() == 1


def test_evaluate_board_zero():
    state = State()
    state.board = [["X", "O", "X"], ["O", "X", "O"], ["O", "X", "O"]]
    assert state.evaluate_board() == 0
