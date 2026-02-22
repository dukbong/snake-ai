import pygame
import random
from collections import namedtuple
from enum import Enum
import numpy as np

_pygame_initialized = False
_font = None


def _ensure_pygame():
    global _pygame_initialized, _font
    if not _pygame_initialized:
        pygame.init()
        _font = pygame.font.SysFont('arial', 25)
        _pygame_initialized = True


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)

BLOCK_SIZE = 20
SPEED = 40


class SnakeGameAI:
    def __init__(self, w=640, h=480, render=True):
        self.w = w
        self.h = h
        self.render = render
        if render:
            _ensure_pygame()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption('Snake AI')
            self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w // 2, self.h // 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y),
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        while True:
            x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            self.food = Point(x, y)
            if self.food not in self.snake:
                break

    def play_step(self, action):
        self.frame_iteration += 1

        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

        # 이동 전 head→food Manhattan distance 저장
        dist_before = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)

        self._move(action)
        self.snake.insert(0, self.head)

        # 충돌 체크 (terminal: 진짜 종료)
        if self.is_collision():
            return -10, True, False, self.score

        # 타임아웃 체크 (truncated: 인위적 제한, V(s)로 bootstrap)
        if self.frame_iteration > 100 * len(self.snake):
            return 0, True, True, self.score

        reward = 0
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            # 이동 후 거리 비교하여 방향성 보상 부여
            dist_after = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
            if dist_after < dist_before:
                reward = 0.1
            else:
                reward = -0.1

        if self.render:
            self._update_ui()
            self.clock.tick(SPEED)

        return reward, False, False, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = _font.render(f'Score: {self.score}', True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # action: [직진, 우회전, 좌회전]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # 직진
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # 우회전
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # 좌회전

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def get_grid_state(self):
        rows = self.h // BLOCK_SIZE  # 24
        cols = self.w // BLOCK_SIZE  # 32
        grid = np.zeros((7, rows, cols), dtype=np.uint8)

        # 채널 0: 몸통 (머리 제외) — numpy fancy indexing
        if len(self.snake) > 1:
            pts = np.array([(p.y // BLOCK_SIZE, p.x // BLOCK_SIZE) for p in self.snake[1:]])
            valid = (pts[:, 0] >= 0) & (pts[:, 0] < rows) & (pts[:, 1] >= 0) & (pts[:, 1] < cols)
            grid[0, pts[valid, 0], pts[valid, 1]] = 1

        # 채널 1: 머리
        r = self.head.y // BLOCK_SIZE
        c = self.head.x // BLOCK_SIZE
        if 0 <= r < rows and 0 <= c < cols:
            grid[1, r, c] = 1

        # 채널 2: 먹이
        r = self.food.y // BLOCK_SIZE
        c = self.food.x // BLOCK_SIZE
        if 0 <= r < rows and 0 <= c < cols:
            grid[2, r, c] = 1

        # 채널 3~6: 방향 one-hot (RIGHT/LEFT/UP/DOWN), 전체 그리드에 broadcast
        if self.direction == Direction.RIGHT:
            grid[3, :, :] = 1
        elif self.direction == Direction.LEFT:
            grid[4, :, :] = 1
        elif self.direction == Direction.UP:
            grid[5, :, :] = 1
        elif self.direction == Direction.DOWN:
            grid[6, :, :] = 1

        return grid
