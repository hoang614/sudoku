from random import sample



grid = [[0] * 9 for _ in range(9)]
list_cell_pos = [(row, col) for row in range(9) for col in range(9)]
goal_state = []
# Hàm sinh trạng thái kết thúc
def generate_cells(cell_index):
    if cell_index == len(list_cell_pos):
        return grid
    list_nums = sample(range(1, 10), 9)
    cell_row, cell_col = list_cell_pos[cell_index]
    for i in list_nums:
        if is_valid((cell_row, cell_col), i, grid):
            grid[cell_row][cell_col] = i
            result = generate_cells(cell_index + 1)
            if result is not None:
                return result
            grid[cell_row][cell_col] = 0
    return None
# Hàm sinh trạng thái khởi đầu
def generate_sudoku_puzzle(n):
    global goal_state
    # Tạo bảng Sudoku hoàn chỉnh
    goal_state = generate_cells(0)
    # Tạo trạng thái khởi đầu với một số ô trống
    initial_state =  [[goal_state[row][col] for col in range(9)] for row in range(9)]
    for pos in sample(range(81), n):
            initial_state[pos // 9][pos % 9] = 0
    return initial_state
# Hàm kiểm tra tính hợp lệ
def is_valid(position, num, grid):
    for col in range(9):
        if grid[position[0]][col] == num:
            return False

    for row in range(9):
        if grid[row][position[1]] == num:
            return False

    box_row = position[0] // 3 * 3
    box_col = position[1] // 3 * 3
    for row in range(3):
        for col in range(3):
            if grid[box_row + row][box_col + col] == num:
                return False

    return True
