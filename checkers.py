from copy import deepcopy
import math
from time import process_time
import multiprocessing
from multiprocessing import Manager

with open('./input.txt', 'r') as f:
    lines = f.readlines()

mode = lines[0].split()[0]
color = lines[1].split()[0]
color = color.lower()
time_left = float(lines[2][:-1])
matrix = [row.strip('\n') for row in lines[3:]]

WIN_SCORE = 10000
LOSE_SCORE = -10000


class Piece:
    def __init__(self, x, y, color, crowned):
        self.x = x
        self.y = y
        self.color = color
        self.crowned = crowned


class Board:
    def __init__(self, matrix, color):
        self.matrix = [[None]*8 for i in range(8)]
        self.color = color

        for i in range(8):
            for j in range(8):
                if matrix[i][j] == '.':
                    continue

                if matrix[i][j] == 'w':
                    self.matrix[i][j] = Piece(i, j, "white", False)
                elif matrix[i][j] == 'W':
                    self.matrix[i][j] = Piece(i, j, "white", True)
                elif matrix[i][j] == 'b':
                    self.matrix[i][j] = Piece(i, j, "black", False)
                elif matrix[i][j] == 'B':
                    self.matrix[i][j] = Piece(i, j, "black", True)

        if self._all_are_kings():
            self.eval = self._sum_of_dist
        else:
            self.eval = self._piece_and_board

    def _all_are_kings(self):
        for i in range(8):
            for j in range(8):
                if self.matrix[i][j] != None and self.matrix[i][j].crowned == False:
                    return False
        return True

    def _piece_and_board(self):
        score = 0
        num = 0
        for i in range(8):
            for j in range(8):
                p = self.matrix[i][j]
                if p == None:
                    continue
                num += 1

                if p.color == 'white' and p.crowned == True:
                    score += 10
                elif p.color != 'white' and p.crowned == True:
                    score -= 10
                elif p.color == 'white' and p.x < 4:
                    score += 7
                elif p.color == 'white' and p.x >= 4:
                    score += 5
                elif p.color != 'white' and p.x < 4:
                    score -= 5
                elif p.color != 'white' and p.x >= 4:
                    score -= 7

        if self.color == 'white':
            return score/num
        else:
            return -score/num

    def _sum_of_dist(self):
        score = 0
        my_pieces = [
            x for row in self.matrix for x in row if x != None and x.color == self.color]
        op_pieces = [
            x for row in self.matrix for x in row if x != None and x.color != self.color]

        for p in my_pieces:
            for op in op_pieces:
                score += math.sqrt((p.x-op.x)**2+(p.y-op.y)**2)

        if len(my_pieces) >= len(op_pieces):
            return 5000-score
        else:
            return score

    def raw_board(self):
        raw = []
        for row in self.matrix:
            tmp = []
            for p in row:
                if p == None:
                    tmp.append('#')
                elif p.color == 'white':
                    tmp.append('w')
                elif p.color == 'black':
                    tmp.append('b')
            raw.append(tmp)

        return raw

    def _legal_location(self, x, y):
        if x < 0 or y < 0 or x > 7 or y > 7:
            return False
        else:
            return True

    def _legal_moves(self, piece, jump_only=False):
        x = piece.x
        y = piece.y
        if piece.color == 'black':
            adversarial_color = 'white'
        else:
            adversarial_color = 'black'

        white_possible_moves = [(x-1, y+1), (x-1, y-1)]
        white_possible_jumps = [(x-2, y+2), (x-2, y-2)]
        black_possible_moves = [(x+1, y+1), (x+1, y-1)]
        black_possible_jumps = [(x+2, y+2), (x+2, y-2)]

        king_possible_moves = white_possible_moves+black_possible_moves
        king_possible_jumps = white_possible_jumps+black_possible_jumps

        if piece.crowned == False and piece.color == "white":
            possible_moves = white_possible_moves
            possible_jumps = white_possible_jumps
        elif piece.crowned == False and piece.color == "black":
            possible_moves = black_possible_moves
            possible_jumps = black_possible_jumps
        else:
            possible_moves = king_possible_moves
            possible_jumps = king_possible_jumps

        possible_moves = [
            move for move in possible_moves if self._legal_location(move[0], move[1])]
        possible_jumps = [
            jump for jump in possible_jumps if self._legal_location(jump[0], jump[1])]

        legal_jumps = []
        for final_x, final_y in possible_jumps:
            via_x = (x+final_x)//2
            via_y = (y+final_y)//2

            if self.matrix[via_x][via_y] != None and self.matrix[via_x][via_y].color == adversarial_color and self.matrix[final_x][final_y] == None:
                legal_jumps.append((final_x, final_y))

        if len(legal_jumps) != 0 or jump_only:
            return legal_jumps

        legal_moves = []
        for final_x, final_y in possible_moves:
            if self.matrix[final_x][final_y] == None:
                legal_moves.append((final_x, final_y))

        return legal_moves

    def all_legal_moves(self, turn):
        pieces = [p for row in self.matrix for p in row if p !=
                  None and p.color == turn]
        all_possible_jumps = []
        can_jump = False
        for p in pieces:
            jumps = self._legal_moves(p, True)
            if len(jumps) != 0:
                can_jump = True
                all_possible_jumps.append((p, jumps))

        if can_jump:
            return all_possible_jumps, True

        all_possible_moves = []
        for p in pieces:
            moves = self._legal_moves(p, False)
            if len(moves) != 0:
                all_possible_moves.append((p, moves))

        return all_possible_moves, False

    def _jump_results(self, piece, jump):
        results = []
        seq = []

        new_board = deepcopy(self)
        new_piece = new_board._do_single_action(piece, jump)

        if piece.crowned == False and new_piece.crowned == True:
            return [new_board], [[jump]]
        new_legal_jumps = new_board._legal_moves(new_piece, jump_only=True)
        if len(new_legal_jumps) == 0:
            return [new_board], [[jump]]

        for new_legal_jump in new_legal_jumps:
            results, sub_seqs = new_board._jump_results(
                new_piece, new_legal_jump)

            for s in sub_seqs:
                s.append(jump)
                seq.append(s)

        return results, seq

    def _do_single_action(self, piece, action):
        from_x, from_y = piece.x, piece.y
        to_x, to_y = action

        new_piece = deepcopy(piece)
        new_piece.x, new_piece.y = action
        if (new_piece.color == 'white' and to_x == 0) or (new_piece.color == 'black' and to_x == 7):
            new_piece.crowned = True

        self.matrix[to_x][to_y] = new_piece
        self.matrix[from_x][from_y] = None

        if abs(from_x - to_x) != 1:
            via_x = (from_x+to_x)//2
            via_y = (from_y+to_y)//2

            self.matrix[via_x][via_y] = None

        return new_piece

    def action_results(self, piece, action, is_jump):
        if is_jump == False:
            board = deepcopy(self)
            board._do_single_action(piece, action)

            return [board], [action]

        else:
            return self._jump_results(piece, action)

    def game_ended(self, turn):
        pieces = [p for row in self.matrix for p in row if p !=
                  None and p.color == turn]
        for p in pieces:
            if len(self._legal_moves(p, jump_only=False)) != 0:
                return False

        return True


def min_value(board, turn, depth, alpha, beta):
    if board.game_ended(turn):
        return WIN_SCORE, None, None
    if depth == 0:
        return board.eval(), None, None

    best_value = math.inf
    best_action = None
    best_piece = None

    next_turn = "white" if turn == "black" else "black"

    actions, is_jump = board.all_legal_moves(turn)

    for piece, action in actions:  # [(piece, [a1,a2,...])]
        for a in action:
            new_boards, action_seq = board.action_results(
                piece, a, is_jump)  # ([new_boards], [move] or [[jump]])

            for i, new_board in enumerate(new_boards):
                value, _, _ = max_value(new_board, next_turn,
                                        depth-1, alpha, beta)
                if value < best_value:
                    best_value = value
                    best_action = action_seq[i]
                    best_piece = piece

                if best_value <= alpha:
                    return best_value, best_piece, best_action
                beta = min(value, beta)

    return best_value, best_piece, best_action


def max_value(board, turn, depth, alpha, beta):
    if board.game_ended(turn):
        return LOSE_SCORE, None, None
    if depth == 0:
        return board.eval(), None, None

    best_value = -math.inf
    best_action = None
    best_piece = None

    next_turn = "white" if turn == "black" else "black"

    actions, is_jump = board.all_legal_moves(turn)

    for piece, action in actions:  # [(piece, [a1,a2,...])]
        for a in action:
            new_boards, action_seq = board.action_results(
                piece, a, is_jump)  # ([new_boards], [move] or [[jump]])

            for i, new_board in enumerate(new_boards):
                value, _, _ = min_value(new_board, next_turn,
                                        depth-1, alpha, beta)
                if value > best_value:
                    best_value = value
                    best_action = action_seq[i]
                    best_piece = piece

                if best_value >= beta:
                    return best_value, best_piece, best_action

                alpha = max(best_value, alpha)

    return best_value, best_piece, best_action


def output(from_index, action):
    def to_algebraic_notation(a):
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        row = 8-a[0]
        col = alphabet[a[1]]

        return str(col)+str(row)

    f = open('./output.txt', 'w')

    if type(action) == list:
        action.reverse()
        for a in action:
            f.writelines('J ' + to_algebraic_notation(from_index) +
                         ' '+to_algebraic_notation(a)+'\n')
            from_index = a
    else:
        f.writelines('E ' + to_algebraic_notation(from_index) +
                     ' ' + to_algebraic_notation(action)+'\n')

    f.seek(f.tell()-1, 0)
    f.truncate()
    f.close()


def minimax_step(board, color, depth, alpha, beta, dic):
    s = process_time()
    _, piece, action = max_value(board, color, depth, alpha, beta)
    dic['result'] = (piece, action)
    dic['iter_time'] = process_time()-s


if __name__ == '__main__':

    s = process_time()

    board = Board(matrix, color)
    if mode == 'SINGLE':
        value, piece, action = max_value(
            board, color, 1, -math.inf, math.inf)
        output((piece.x, piece.y), action)
    else:
        if time_left > 200:
            step_time = 20
        elif time_left > 100:
            step_time = 10
        elif time_left > 50:
            step_time = 5
        elif time_left > 30:
            step_time = 3
        else:
            step_time = 1
            step_time = min(step_time, time_left/2)

        # step_time=15
        max_depth = 12
        dic = Manager().dict()

        step_time *= 0.9
        total_iter_time = 0

        print('step time: '+str(step_time))

        for i in range(1, max_depth+1):
            start_time = process_time()
            proc = multiprocessing.Process(target=minimax_step, args=(
                board, color, i, -math.inf, math.inf, dic))
            proc.start()
            proc.join(step_time-process_time()-total_iter_time)
            if proc.is_alive():
                proc.terminate()
                piece, action = dic['result']

                output((piece.x, piece.y), action)
                print('timeout: max depth is %d' % (i-1))
                exit()

            iter_time = dic['iter_time']
            total_iter_time += iter_time
            print(iter_time)

            if step_time-process_time()-total_iter_time < 5*iter_time:
                piece, action = dic['result']
                output((piece.x, piece.y), action)
                print('too little time left: max depth is %d, run time is %f' %
                      (i, process_time()+total_iter_time))
                exit()

        piece, action = dic['result']
        output((piece.x, piece.y), action)
        print('max depth limit: max depth is %d, run time is %f' %
              (i, process_time()+total_iter_time))
        exit()