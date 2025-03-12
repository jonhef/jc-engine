import pygame
import os
import random
from model.predict import ChessEngine
import chess
import time

# Инициализация Pygame
pygame.init()

engine = ChessEngine('checkpoint.pth')

WIDTH, HEIGHT = 800, 800
SQUARE_SIZE = WIDTH // 8
SPRITE_SIZE = 16
SPACING = 4

# Порядок фигур в спрайтшите
PIECES_ORDER = ['P', 'N', 'R', 'B', 'Q', 'K']

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
HIGHLIGHT_COLOR = (100, 200, 100, 100)

# Загрузка и подготовка изображений
def load_pieces():
    images = {}
    SPRITE_WIDTH = 12
    SPRITE_HEIGHT = 16  # Высота 16 пикселей
    SPACING = 4
    
    # Порядок фигур в спрайтшите
    PIECES_ORDER = ['P', 'N', 'R', 'B', 'Q', 'K']
    
    def extract_sprites(sheet):
        sprites = []
        x = 0
        for _ in PIECES_ORDER:
            # Вырезаем спрайт с учётом новой высоты
            sprite = sheet.subsurface(pygame.Rect(x, 0, SPRITE_WIDTH, SPRITE_HEIGHT))
            sprites.append(sprite)
            x += SPRITE_WIDTH + SPACING
        return sprites
    
    # Загрузка и обработка белых фигур
    white_sheet = pygame.image.load('WhitePieces.png').convert_alpha()
    white_sprites = extract_sprites(white_sheet)
    
    # Загрузка и обработка чёрных фигур
    black_sheet = pygame.image.load('BlackPieces.png').convert_alpha()
    black_sprites = extract_sprites(black_sheet)
    
    # Масштабирование с сохранением пропорций и центрированием
    for i, piece in enumerate(PIECES_ORDER):
        # Рассчитываем масштаб для белых фигур
        original_width, original_height = SPRITE_WIDTH, SPRITE_HEIGHT
        ratio = min(SQUARE_SIZE/original_width, SQUARE_SIZE/original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        
        # Создаём поверхность для белой фигуры
        white_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        scaled_white = pygame.transform.scale(white_sprites[i], (new_width, new_height))
        x_offset = (SQUARE_SIZE - new_width) // 2
        y_offset = (SQUARE_SIZE - new_height) // 2
        white_surface.blit(scaled_white, (x_offset, y_offset))
        images['w' + piece] = white_surface
        
        # Аналогично для чёрных фигур
        black_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        scaled_black = pygame.transform.scale(black_sprites[i], (new_width, new_height))
        black_surface.blit(scaled_black, (x_offset, y_offset))
        images['b' + piece] = black_surface
    
    return images


# Класс шахматной доски
class ChessBoard:
    def __init__(self):
        self.board = self.create_initial_board()
        self.selected_piece = None
        self.current_player = 'white'
        self.move_history = []
        
    def create_initial_board(self):
        board = [[None for _ in range(8)] for _ in range(8)]
        
        # Белые фигуры
        pieces = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        for i in range(8):
            board[0][i] = 'w' + pieces[i]
            board[1][i] = 'wP'
        
        # Чёрные фигуры
        for i in range(8):
            board[7][i] = 'b' + pieces[i]
            board[6][i] = 'bP'
            
        return board

    
    def generate_fen(self):
        """Генерация FEN строки текущей позиции"""
        fen = []
        empty = 0
        
        for row in self.board:
            for piece in row:
                if piece is None:
                    empty += 1
                else:
                    if empty > 0:
                        fen.append(str(empty))
                        empty = 0
                    fen.append(piece[1].upper() if piece[0] == 'w' else piece[1].lower())
            if empty > 0:
                fen.append(str(empty))
                empty = 0
            fen.append('/')
        
        # Убираем последний слэш и добавляем остальные параметры
        print(fen)
        print(str(''.join(fen[:-1]) + ' w KQkq - 0 1'))
        return str(''.join(fen[:-1]) + ' w KQkq - 0 1')

    
    def get_possible_moves(self, row, col):
        piece = self.board[row][col]
        if not piece or piece[0] != 'w':
            return []

        moves = []
        piece_type = piece[1]

        # Общие правила движения для белых фигур
        if piece_type == 'P':  # Пешка
            # Движение вперед
            if row < 7 and self.board[row+1][col] is None:
                moves.append((row+1, col))
                # Начальный двойной ход
                if row == 1 and self.board[row+2][col] is None:
                    moves.append((row+2, col))
            
            # Взятие фигур
            for dx in [-1, 1]:
                if 0 <= col+dx < 8 and row < 7:
                    target = self.board[row+1][col+dx]
                    if target and target[0] == 'b':
                        moves.append((row+1, col+dx))

        elif piece_type == 'N':  # Конь
            knight_moves = [
                (row+2, col+1), (row+2, col-1),
                (row-2, col+1), (row-2, col-1),
                (row+1, col+2), (row+1, col-2),
                (row-1, col+2), (row-1, col-2)
            ]
            for r, c in knight_moves:
                if 0 <= r < 8 and 0 <= c < 8:
                    if self.board[r][c] is None or self.board[r][c][0] == 'b':
                        moves.append((r, c))

        elif piece_type == 'B':  # Слон
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    if self.board[r][c] is None:
                        moves.append((r, c))
                    elif self.board[r][c][0] == 'b':
                        moves.append((r, c))
                        break
                    else:
                        break
                    r += dr
                    c += dc

        elif piece_type == 'R':  # Ладья
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                r, c = row + dr, col + dc
                while 0 <= r < 8 and 0 <= c < 8:
                    if self.board[r][c] is None:
                        moves.append((r, c))
                    elif self.board[r][c][0] == 'b':
                        moves.append((r, c))
                        break
                    else:
                        break
                    r += dr
                    c += dc

        elif piece_type == 'Q':  # Ферзь
            # Комбинация слона и ладьи
            self.board[row][col] = 'B' + piece[1]
            moves.extend(self.get_possible_moves(row, col))
            self.board[row][col] = 'R' + piece[1]
            moves.extend(self.get_possible_moves(row, col))
            self.board[row][col] = piece

        elif piece_type == 'K':  # Король
            king_moves = [
                (row+1, col), (row-1, col),
                (row, col+1), (row, col-1),
                (row+1, col+1), (row+1, col-1),
                (row-1, col+1), (row-1, col-1)
            ]
            for r, c in king_moves:
                if 0 <= r < 8 and 0 <= c < 8:
                    if self.board[r][c] is None or self.board[r][c][0] == 'b':
                        moves.append((r, c))

        return moves
    
    def move_piece(self, from_pos, to_pos):
        from_row, from_col = from_pos
        to_row, to_col = to_pos

        # Проверка на превращение пешки
        piece = self.board[from_row][from_col]
        if piece[1] == 'P' and to_row == 7:
            # Автоматическое превращение в ферзя (можно добавить выбор фигуры)
            piece = 'wQ'

        self.board[to_row][to_col] = piece
        self.board[from_row][from_col] = None
        self.current_player = 'black' if self.current_player == 'white' else 'white'
        self.move_history.append((from_pos, to_pos))

# API для получения хода чёрных (заглушка)
def get_black_move(fen):
    # Простой алгоритм для демонстрации
    return engine.predict_move(fen)



# Основной класс игры
class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Шахматы")
        self.images = load_pieces()
        self.chess_board = ChessBoard()
        self.running = True
        
    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                pygame.draw.rect(self.screen, color, 
                               (col*SQUARE_SIZE, row*SQUARE_SIZE,
                                SQUARE_SIZE, SQUARE_SIZE))
                
                # Отрисовка фигур
                piece = self.chess_board.board[row][col]
                if piece:
                    self.screen.blit(self.images[piece],
                                    (col*SQUARE_SIZE, row*SQUARE_SIZE))
                    
                # Подсветка выбранной фигуры
                if self.chess_board.selected_piece == (row, col):
                    surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    surface.fill(HIGHLIGHT_COLOR)
                    self.screen.blit(surface, (col*SQUARE_SIZE, row*SQUARE_SIZE))
    
    def handle_click(self, pos):
        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE

        if self.chess_board.current_player == 'white':
            if self.chess_board.selected_piece is None:
                # Выбор фигуры игрока
                piece = self.chess_board.board[row][col]
                if piece and piece[0] == 'w':
                    self.chess_board.selected_piece = (row, col)
            else:
                # Попытка сделать ход
                from_row, from_col = self.chess_board.selected_piece
                possible_moves = self.chess_board.get_possible_moves(from_row, from_col)
                
                if (row, col) in possible_moves:
                    self.chess_board.move_piece((from_row, from_col), (row, col))
                    
                    # Ход черных
                    fen = self.chess_board.generate_fen()
                    black_move = get_black_move(fen)
                    if black_move:
                        # Конвертация UCI нотации
                        from_sq = black_move[:2]
                        to_sq = black_move[2:]
                        from_col = ord(from_sq[0]) - ord('a')
                        from_row = 8 - int(from_sq[1])
                        to_col = ord(to_sq[0]) - ord('a')
                        to_row = 8 - int(to_sq[1])
                        
                        # Проверка валидности хода
                        if self.chess_board.board[from_row][from_col] and \
                           self.chess_board.board[from_row][from_col][0] == 'b':
                            self.chess_board.move_piece((from_row, from_col), (to_row, to_col))

                self.chess_board.selected_piece = None
        time.sleep(1)
        if self.chess_board.current_player == 'black':
            # Получаем FEN и ход от API
            fen = self.chess_board.generate_fen()
            black_move = get_black_move(fen)
            
            print(black_move)
            if black_move:
                # Конвертируем UCI нотацию в координаты
                from_col = ord(black_move[0]) - ord('a')
                from_row = 8 - int(black_move[1])
                to_col = ord(black_move[2]) - ord('a')
                to_row = 8 - int(black_move[3])
                
                self.chess_board.move_piece((from_row, from_col), (to_row, to_col))
        
    
    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(pygame.mouse.get_pos())
            
            self.draw_board()
            pygame.display.flip()
        
        pygame.quit()

if __name__ == "__main__":
    game = Game()
    game.run()