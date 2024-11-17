import pygame
import numpy
import random
import time
import sys
import generate

pygame.init()
pygame.font.init()
random.seed()
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
GREY = (169, 169, 169)

PUZZLE_SIZE = 9
BOX_SIZE = 3


# Lớp Grid đại diện cho bảng Sudoku
class Grid:
    values = [[0 for _ in range(PUZZLE_SIZE)] for _ in range(PUZZLE_SIZE)]

    def __init__(self, rows, cols, width, height):
        self.rows = rows
        self.cols = cols
        self.cubes = [[Cube(self.values[i][j], i, j, width, height) for j in range(cols)] for i in range(rows)]
        self.width = width
        self.height = height
        self.model = None
        self.selected = None

    def update_model(self):
        self.model = [[self.cubes[i][j].value for j in range(self.cols)] for i in range(self.rows)]

    def place(self, val):
        row, col = self.selected
        self.values[row][col] = val
        self.cubes[row][col].set(val)
        self.update_model()

    def draw(self, win):
        gap = self.width / 9
        for i in range(self.rows + 1):
            if i % 3 == 0 and i != 0:
                thick = 4  
            else:
                thick = 1
            pygame.draw.line(win, BLACK, (0, i * gap), (self.width, i * gap), thick)
            pygame.draw.line(win, BLACK, (i * gap, 0), (i * gap, self.height), thick)

        for i in range(self.rows):
            for j in range(self.cols):
                self.cubes[i][j].draw(win)

    def select(self, row, col):
        # Reset tất cả các ô đã chọn
        for i in range(self.rows):
            for j in range(self.cols):
                self.cubes[i][j].selected = False

        self.cubes[row][col].selected = True
        self.selected = (row, col)

    def clear(self):
        row, col = self.selected
        self.values[row][col] = 0
        self.cubes[row][col].set(0)

    def click(self, pos):
        if pos[0] < self.width and pos[1] < self.height:
            gap = self.width / 9
            x = pos[0] // gap
            y = pos[1] // gap
            return int(y), int(x)
        else:
            return None

    def is_finished(self):
        for i in range(self.rows):
            for j in range(self.cols):
                if self.cubes[i][j].value == 0:
                    return False
        return True

    def update_result(self):
        self.cubes = [[Cube(self.values[i][j], i, j, self.width, self.height) for j in range(self.cols)] for i in
                      range(self.rows)]
        self.update_model()

# Lớp Cube đại diện cho 1 ô trong bảng Sudoku
class Cube:
    rows = PUZZLE_SIZE
    cols = PUZZLE_SIZE

    def __init__(self, value, row, col, width, height):
        self.value = value
        self.temp = 0
        self.row = row
        self.col = col
        self.width = width
        self.height = height
        self.selected = False

    def draw(self, win):
        font = pygame.font.SysFont("comicsans", 40)

        gap = self.width / 9
        x = self.col * gap
        y = self.row * gap

        if self.value != 0:
            text = font.render(str(self.value), True, BLACK)
            win.blit(text, (x + (gap / 2 - text.get_width() / 2), y + (gap / 2 - text.get_height() / 2)))

        if self.selected:
            pygame.draw.rect(win, RED, (x, y, gap, gap), 3)

    def set(self, val):
        self.value = val
    


def win_draw(board, win):
    win.fill(WHITE)
    board.draw(win)



# SOLVING

#check row
def is_row_duplicate(board, pos, num):
    for i in range(0, PUZZLE_SIZE):
        if board[pos[0]][i] == num:
            return True
    return False

# check collume
def is_col_duplicate(board, pos, num):
    for i in range(0, PUZZLE_SIZE):
        if board[i][pos[1]] == num:
            return True
    return False

# Check box 3x3
def is_block_duplicate(board, pos, num):
    box_x = pos[1] // 3
    box_y = pos[0] // 3
    
    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x * 3, box_x*3 + 3):
            if board[i][j] == num:
                return True
    return False

# check all
def valid(board, num, pos):
    
    if (is_row_duplicate(board, pos, num) 
        or is_col_duplicate(board, pos, num) 
        or is_block_duplicate(board, pos, num)):
        return False
    return True


class Candidate(object):
    def __init__(self):
        self.values = [([0]*PUZZLE_SIZE) for i in range(PUZZLE_SIZE)]
        self.fitness = 0.0
        return
    
    def complete(self):
        if self.fitness == 1.0:
            return True
        
    def update_fitness(self):
        col_sum = 0
        block_sum = 0
        for col in range(PUZZLE_SIZE):
            col_values = [self.values[row][col] for row in range(PUZZLE_SIZE)] 
            col_count = len(set(col_values))
            col_sum += col_count / 9.0
        col_fitness = col_sum / 9.0     # trung bình fitness của tất cả các cột

        for i in range(0,PUZZLE_SIZE,3):
            for j in range(0,PUZZLE_SIZE,3):
                block_values = [self.values[i+x][j+y] 
                                for x in range(3)
                                for y in range(3)]
                block_count = len(set(block_values))
                block_sum += block_count / (PUZZLE_SIZE*1.0)
        # trung bình fitness của tất cả các khối 3x3
        block_fitness = block_sum / (PUZZLE_SIZE*1.0)   
        # fitness tổng thể
        if col_fitness < 1.0 or block_fitness < 1.0:
            self.fitness = col_fitness * block_fitness
        else:
            self.fitness = 1.0
        return
    # đột biến bằng cách hoán đổi 2 vị trí mà có giá trị trống trong ma trận khởi đầu
    def mutate(self, mutation_rate, given):
        r = random.uniform(0,1.0000000000001)
        success = False
        if r < mutation_rate:
            while not success:
                row1 = random.randint(0,8)
                row2 = row1
                col1 = random.randint(0,8)
                col2 = random.randint(0,8)
                while col1 == col2:
                    col2 = random.randint(0,8)
                if given[row1][col1] == 0 and given[row1][col2] == 0:
                    if valid(given,self.values[row1][col2], (row1,col1)) and \
                        valid(given,self.values[row2][col1],(row2,col2)):
                        temp = self.values[row1][col1]
                        self.values[row1][col1] = self.values[row2][col2]
                        self.values[row2][col2] = temp
                        success = True
        return success
    
class Population:
    def __init__(seft):
        seft.candidates = []
        return
        
    def seed(self,No_candidate, given):
        helper = Candidate()
        helper.values = [[[] for _ in range(9)] for _ in range(9)]  
        seeding = True
        while seeding:     
            for row in range(9):
                for col in range(9):
                    if given[row][col] == 0:
                        helper.values[row][col] = [val for val in range(1,10) if valid(given,val,(row,col))]
                    else:
                        helper.values[row][col] = [given[row][col]]
            for _ in range(0,No_candidate):
                candidate = Candidate()
                for i in range(9):
                    row = [0 for i in range(9)]
                    for j in range(9):
                        if given[i][j] != 0:
                            row[j] = given[i][j]
                        else:
                            row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j])-1)]
                    while (len(list(set(row))) != 9):
                        for j in range(9):
                            if given[i][j] == 0:
                                row[j] = helper.values[i][j][random.randint(0, len(helper.values[i][j])-1)]
                    candidate.values[i] = row
                self.candidates.append(candidate)  
            self.update_fitness()
            seeding =False
        return
    def update_fitness(self):
        for candidate in self.candidates:
            candidate.update_fitness()
        return
    def sort(self):
        self.candidates.sort(key=lambda x: x.fitness, reverse=True)
            

def compete(candidates):
        # chọn 2 cá thể để cạnh tranh nhau
        candidate1 = random.choice(candidates)
        candidate2 = random.choice(candidates)
    
        fitness1 = candidate1.fitness
        fitness2 = candidate2.fitness
    
        if fitness1 > fitness2:
            fittest, weakest = candidate1, candidate2
        else:
            fittest, weakest = candidate2, candidate1

        selection_rate = 0.85
        r = random.uniform(0, 1.00000000000001)
        if r < selection_rate:
            return fittest
        else:
            return weakest

class CycleCrossover(object):

    def __init__(self):
        return
     # Tạo 2 con bằng cách lai từ 2 bố mẹ
    def crossover(self, parent1, parent2, crossover_rate):
       
        child1 = Candidate()
        child2 = Candidate()
        
        # Sao chép từ gen của bố mẹ
        child1.values = list(parent1.values)
        child2.values = list(parent2.values)      
        r = random.uniform(0, 1.00000000000001)
        if r < crossover_rate:
            crossover_point1, crossover_point2 = sorted(random.sample(range(10),2))
            for i in range(crossover_point1, crossover_point2):
                child1.values[i], child2.values[i] = self.crossover_rows(child1.values[i], child2.values[i])
        return child1, child2

    def crossover_rows(self, row1, row2): 
        child_row1, child_row2 = [0] * 9, [0] * 9

        remaining = list(range(1,10))
        cycle = 0
        while((0 in child_row1) and (0 in child_row2)):
            index = self.find_unused(row1, remaining)
            start_value = row1[index]
            remaining.remove(start_value)
            next_value = row2[index]
            while True:
                if cycle % 2 == 0:
                    child_row1[index], child_row2[index] = row1[index], row2[index]
                else:
                    child_row1[index], child_row2[index] = row2[index], row1[index]
                if next_value == start_value:
                    break
                index = self.find_value(row1, next_value)
                next_value = row2[index]
            cycle += 1
        return child_row1, child_row2  
    
    def find_unused(self, parent_row, remaining):
        for i in range(0, len(parent_row)):
            if parent_row[i] in remaining:
                return i

    def find_value(self, parent_row, value):
        for i in range(0, len(parent_row)):
            if parent_row[i] == value:
                return i





def solve(board):
    tol_gen = 0
    solution = None
    Nc = 1000  # kích thước quần thể
    Ne = int(0.05*Nc)  # số lượng cá thể tốt nhất
    Ng = 10000  # Số thế hệ tối đa
    Nm = 0  # Số lần đột biến
    # Tham số đột biến
    phi = 0 # tỷ lệ thành công của đột biến
    sigma = 1 # tham số điều chỉnh tỷ lệ đột biến
    mutation_rate = 0.06 # tỷ lệ đột biến ban đầu
    mutation_rate_elite = 0.5 # tỷ lệ đột biến cho các cá thể tốt nhất
    # Khởi tạo quần thể
    population = Population()
    population.seed(Nc, board)
    best_fitness = 0
    count_best_fit_unchange = 0
    count_bound = 0

    for generation in range(Ng):
        print(f"Generation {generation}")

        population.update_fitness() # cập nhật fitness
        population.sort()   # sắp xếp quần thể theo fitness

        if best_fitness == population.candidates[0].fitness:
            count_best_fit_unchange += 1
        else:
            count_best_fit_unchange = 0
            if best_fitness < population.candidates[0].fitness:
                best_fitness = population.candidates[0].fitness
        print(f"Fitness: {best_fitness}")
        # đột biến cá thể tốt nhất nếu fitness k cải thiện sau 1000 thế hệ
        if count_best_fit_unchange >= 1000:

            for e in range(Ne):
                elite = population.candidates[e]
                old_fitness = elite.fitness
                success = elite.mutate(mutation_rate_elite, board)
                if success:
                    elite.update_fitness()
                    Nm += 1
                    if elite.fitness > old_fitness: 
                        phi = phi + 1
            count_best_fit_unchange = 0

        if best_fitness == 1:   # tìm được giải pháp
            tol_gen += generation
            solution = population.candidates[0].values
            break

        # Tạo quần thể tiếp theo
        next_population = []

        # Chọn cá thể tốt nhất
        for e in range(0, Ne):
            elite = population.candidates[e]
            next_population.append(elite)
        # tạo cá thể mới thông qua mutate và crossover
        for count in range(Ne, Nc, 2):
            parent1 = compete(population.candidates)
            parent2 = compete(population.candidates)

            # Thực hiện Crossover.
            cc = CycleCrossover()
            child1, child2 = cc.crossover(parent1, parent2, crossover_rate=1.0)

            # Đột biến child1.
            child1.update_fitness()
            old_fitness = child1.fitness
            success = child1.mutate(mutation_rate, board)
            child1.update_fitness()
            if success:
                Nm += 1
                if child1.fitness > old_fitness:  
                    phi = phi + 1

            # Đột biến child2.
            child2.update_fitness()
            old_fitness = child2.fitness
            success = child2.mutate(mutation_rate, board)
            child2.update_fitness()
            if success:
                Nm += 1
                if(child2.fitness > old_fitness): 
                    phi = phi + 1

            # Thêm các child vào quần thể tiếp theo
            next_population.append(child1)
            next_population.append(child2)
        if child1.fitness == 1.0:
            return child1.values
        if child2.fitness == 1.0:
            return child2.values
        # cập nhật quần thể hiện tại
        population.candidates = next_population

        # điều chỉnh tỷ lệ đột biến dựa trên tỷ lệ thành công
        if Nm == 0:
            phi = 0
        else:
            phi = phi / Nm

        if phi > 0.2:
            sigma = sigma/0.998
        elif phi < 0.2:
            sigma = sigma*0.998

        mutation_rate = abs(numpy.random.normal(loc=0.0, scale=sigma, size=None))
        Nm = 0
        phi = 0

        # Tái tạo quần thể nếu cần
        up_bound = 2000
        if count_bound >= up_bound:
            print("The population has gone stale. Re-seeding...")
            population.seed(Nc, board)
            sigma = 1
            phi = 0
            Nm = 0
            mutation_rate = 0.06
            count_bound = -1
        count_bound += 1
    if solution == None:
        print('No found solution')
        pygame.quit()
        sys.exit()
    return solution


if __name__=='__main__':
    win = pygame.display.set_mode((550, 550))
    pygame.display.set_caption("Sudoku")
    board = Grid(9, 9, 550, 550)
    key = None
    run = True
    start = time.time() 

    initial_state = generate.generate_sudoku_puzzle(50) # 60 ô trống
    print("Initial state")
    for row in initial_state:
        print(row)
    print("Goal state")
    goal_state = generate.goal_state
    for row in goal_state:
        print(row)
    board.values = [[initial_state[row][col] for col in range(PUZZLE_SIZE)] for row in range(PUZZLE_SIZE)]
    board.update_result() 

    while run:

        win_draw(board,win)
        current_time = round(time.time()-start)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                clicked = board.click(pos)
                if clicked:
                    board.select(clicked[0], clicked[1])
                    key = None
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    key = 1
                elif event.key == pygame.K_2:
                    key = 2
                elif event.key == pygame.K_3:
                    key = 3
                elif event.key == pygame.K_4:
                    key = 4
                elif event.key == pygame.K_5:
                    key = 5
                elif event.key == pygame.K_6:
                    key = 6
                elif event.key == pygame.K_7:
                    key = 7
                elif event.key == pygame.K_8:
                    key = 8
                elif event.key == pygame.K_9:
                    key = 9
                elif event.key == pygame.K_DELETE:
                    board.clear()
                    key = None
                elif event.key == pygame.K_RETURN:
                    start = time.time()
                    board.values = solve(board.values)
                    current_time = round(time.time() - start)
                    print("Time: ",current_time,"s")
                    # print(board.values)
                    board.update_result()
                    win_draw(board, win)
                elif event.key == pygame.K_SPACE:
                    generate.grid = [[0] * 9 for _ in range(9)]
                    generate.goal_state = []
                    initial_state = generate.generate_sudoku_puzzle(50) # 50 ô trống
                    print("Initial state")
                    for row in initial_state:
                        print(row)
                    print("Goal state")
                    goal_state = generate.goal_state
                    for row in goal_state:
                        print(row)
                    board.values = [[initial_state[row][col] for col in range(PUZZLE_SIZE)] for row in range(PUZZLE_SIZE)]
                    board.update_result()
                    win_draw(board, win)

        if board.selected and key!=None:
            board.place(key)
        pygame.display.update()
    pygame.quit()
    sys.exit()

                
        
        

        