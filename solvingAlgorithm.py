
def find_empty(board):
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                return (r,c)
    return None

def valid(board, num, pos):
    for i in range(len(board[0])):
        if board[pos[0]][i] == num and pos[1] != i:
            return False

    for i in range(len(board)):
        if board[i][pos[1]] == num and pos[0] != i:
            return False

    box_x = pos[1] // 3
    box_y = pos[0] // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if board[i][j] == num and (i,j) != pos:
                return False

    return True


def solve(board):
    pos = find_empty(board)
    if not pos:
        return True
    else:
        r,c = pos

    for i in range(1,10):
        if valid(board, i, (r, c)):
            board[r][c] = i

            if solve(board):
                return True
            board[r][c] = 0
    return False


def get_board(bo):
    if solve(bo):
        return bo
    else:
        raise ValueError




