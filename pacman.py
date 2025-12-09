import random
import sys
import heapq
from dataclasses import dataclass
from typing import List

import pygame
from pygame.math import Vector2

# --- Configuration ---------------------------------------------------------

TILE_SIZE = 20
HUD_HEIGHT = 60
FPS = 60
PACMAN_SPEED = 2
GHOST_SPEED = 2
POWER_MODE_DURATION = 6.0
DANGER_RANGE = 4 * TILE_SIZE # New: Pacman flees if a ghost is within 4 tiles

# Colors
BLACK = (0, 0, 0)
BLUE = (33, 33, 255)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
POWER_COLOR = (255, 184, 174)
FRIGHTENED_COLOR = (48, 48, 255)
GHOST_COLORS = [
    (255, 0, 0),
    (255, 105, 180),
    (0, 255, 255),
    (255, 165, 0),
]

# Layout legend:
# 1 = wall, 0 = pellet, o = power pellet, P = Pac-Man start, G = ghost start

LEVEL_LAYOUT = [
  "1111111111111111111111111111",
  "1o000000000111110000000000o1",
  "1001111100011111001111100001",
  "1011111100011111001111111101",
  "1010000000000000000000000101",
  "1010111111111101111111100101",
  "1010000000000000000000000101",
  "1010000111111101111110000101",
  "1000000100000000000010000001",
  "1000000001111111100000000001",
  "1000000000GGGG00000000000001",
  "1000000001111111100000000001",
  "1000000100000000000010000001",
  "1010000111111101111110000101",
  "1010000000000000000000000101",
  "1010111111111101111111110101",
  "1010000000000P00000000000101",
  "1011111100000000001111111001",
  "1001111100111111001111100001",
  "1o000000001111110000000000o1",
  "1111111111111111111111111111",
];

CARDINAL_DIRECTIONS = [
    Vector2(1, 0),
    Vector2(-1, 0),
    Vector2(0, 1),
    Vector2(0, -1),
]

DIRECTION_VECTORS = {
    pygame.K_LEFT: (-1, 0),
    pygame.K_RIGHT: (1, 0),
    pygame.K_UP: (0, -1),
    pygame.K_DOWN: (0, 1),
}


# --- Data objects ----------------------------------------------------------

@dataclass
class Pellet:
    center: Vector2
    power: bool = False

    @property
    def radius(self) -> int:
        return 6 if self.power else 3

    def draw(self, surface: pygame.Surface) -> None:
        color = POWER_COLOR if self.power else WHITE
        pygame.draw.circle(surface, color, (int(self.center.x), int(self.center.y)), self.radius)

    def collides(self, rect: pygame.Rect) -> bool:
        return rect.collidepoint(self.center.x, self.center.y)


class Maze:
    # ... (Maze class is unchanged) ...
    def __init__(self, layout: List[str]) -> None:
        self.tile_size = TILE_SIZE
        self.layout = layout
        self.rows = len(layout)
        self.cols = len(layout[0])
        self.pixel_width = self.cols * self.tile_size
        self.pixel_height = self.rows * self.tile_size
        self.walls: List[pygame.Rect] = []
        self.pellet_blueprint: List[tuple[Vector2, bool]] = []
        self.player_start: Vector2 | None = None
        self.ghost_starts: List[Vector2] = []
        self._parse_layout()

    def _parse_layout(self) -> None:
        for row_idx, row in enumerate(self.layout):
            for col_idx, cell in enumerate(row):
                x = col_idx * self.tile_size
                y = row_idx * self.tile_size
                if cell == "1":
                    self.walls.append(pygame.Rect(x, y, self.tile_size, self.tile_size))
                elif cell == "0":
                    center = Vector2(x + self.tile_size // 2, y + self.tile_size // 2)
                    self.pellet_blueprint.append((center, False))
                elif cell == "o":
                    center = Vector2(x + self.tile_size // 2, y + self.tile_size // 2)
                    self.pellet_blueprint.append((center, True))
                elif cell == "P":
                    self.player_start = Vector2(x, y)
                elif cell == "G":
                    self.ghost_starts.append(Vector2(x, y))

        if self.player_start is None:
            raise ValueError("Layout must include a Pac-Man start tile (P).")
        if not self.ghost_starts:
            raise ValueError("Layout must include at least one ghost start tile (G).")

    def create_pellets(self) -> List[Pellet]:
        return [Pellet(Vector2(pos), power) for pos, power in self.pellet_blueprint]

    def draw(self, surface: pygame.Surface) -> None:
        for wall in self.walls:
            pygame.draw.rect(surface, BLUE, wall, border_radius=4)


class Entity:
    # ... (Entity class is unchanged) ...
    def __init__(self, start_pos: Vector2, color: tuple[int, int, int], speed: int) -> None:
        self.start_pos = Vector2(start_pos)
        self.rect = pygame.Rect(int(self.start_pos.x), int(self.start_pos.y), TILE_SIZE, TILE_SIZE)
        self.direction = Vector2(0, 0)
        self.speed = speed
        self.color = color

    def reset(self) -> None:
        self.rect.topleft = (int(self.start_pos.x), int(self.start_pos.y))
        self.direction.update(0, 0)

    def move(self, walls: List[pygame.Rect]) -> bool:
        if self.direction.length_squared() == 0:
            return False

        dx = int(self.direction.x * self.speed)
        dy = int(self.direction.y * self.speed)
        candidate = self.rect.move(dx, dy)
        if any(candidate.colliderect(wall) for wall in walls):
            return False

        self.rect = candidate
        return True

    def at_tile_center(self) -> bool:
        center_x = (self.rect.centerx - TILE_SIZE // 2) % TILE_SIZE
        center_y = (self.rect.centery - TILE_SIZE // 2) % TILE_SIZE
        return center_x == 0 and center_y == 0


class Pacman(Entity):
    def __init__(self, start_pos: Vector2) -> None:
        super().__init__(start_pos, YELLOW, PACMAN_SPEED)
        self.desired_direction = Vector2(0, 0)
        self.mode = "seek"  # NEW: 'seek' or 'flee'
        self.flee_cooldown = 0.0 # NEW: Timer to prevent rapid mode switching

    def queue_direction(self, direction: Vector2) -> None:
        self.desired_direction = Vector2(direction)

    def can_move(self, direction: Vector2, walls: List[pygame.Rect]) -> bool:
        if direction.length_squared() == 0:
            return False
        dx = int(direction.x * TILE_SIZE)
        dy = int(direction.y * TILE_SIZE)
        candidate = self.rect.move(dx, dy)
        return not any(candidate.colliderect(wall) for wall in walls)

    def find_nearest_pellet(self, pellets: List[Pellet]):
        """Find the nearest pellet to Pac-Man's current position."""
        if not pellets:
            return None

        pacman_pos = Vector2(self.rect.center)
        nearest_pellet = min(pellets, key=lambda p: pacman_pos.distance_squared_to(p.center))
        return nearest_pellet

    def heuristic(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(self, position: tuple[int, int], walls: List[pygame.Rect]) -> List[tuple[int, int]]:
        """Get all valid neighboring tile positions for a given position."""
        neighbors = []
        x, y = position

        for direction in CARDINAL_DIRECTIONS:
            neighbor_x = x + int(direction.x * TILE_SIZE)
            neighbor_y = y + int(direction.y * TILE_SIZE)
            neighbor_rect = pygame.Rect(neighbor_x, neighbor_y, TILE_SIZE, TILE_SIZE)

            if not any(neighbor_rect.colliderect(wall) for wall in walls):
                neighbors.append((neighbor_x, neighbor_y))

        return neighbors

    def astar(self, start, goal, walls: List[pygame.Rect]):
        # ... (A* function is unchanged) ...
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g = {start: 0}
        f = {start: self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]

            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for neighbor in self.get_neighbors(current, walls):
                tentative_g = g[current] + 1

                if neighbor not in g or tentative_g < g[neighbor]:
                    came_from[neighbor] = current
                    g[neighbor] = tentative_g
                    f[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f[neighbor], f[neighbor]))

        return []

    def _path_to_direction(self, start_pos, next_pos) -> Vector2:
        """Helper to convert the first step of a path into a direction vector."""
        dx = next_pos[0] - start_pos[0]
        dy = next_pos[1] - start_pos[1]
        if dx > 0: return Vector2(1, 0)
        if dx < 0: return Vector2(-1, 0)
        if dy > 0: return Vector2(0, 1)
        if dy < 0: return Vector2(0, -1)
        return Vector2(0, 0)

    def is_in_danger(self, danger_ghosts: List['Ghost']) -> bool:
        """NEW: Check if any non-frightened ghost is within the danger range."""
        pacman_pos = Vector2(self.rect.center)
        for ghost in danger_ghosts:
            if pacman_pos.distance_to(ghost.rect.center) < DANGER_RANGE:
                return True
        return False

    def get_escape_direction(self, danger_ghosts: List['Ghost'], walls: List[pygame.Rect]) -> Vector2:
        """NEW: Choose the direction that maximizes distance from the closest danger ghost."""
        if not danger_ghosts:
            return Vector2(0, 0)

        # 1. Find the nearest dangerous ghost
        pacman_pos = Vector2(self.rect.center)
        closest_ghost = min(danger_ghosts, key=lambda g: pacman_pos.distance_squared_to(g.rect.center))
        closest_ghost_pos = Vector2(closest_ghost.rect.center)

        best_score = -float('inf')
        best_direction = Vector2(0, 0)

        # 2. Score all available directions
        available_directions = CARDINAL_DIRECTIONS
        for direction in available_directions:
            # Check if move is blocked by a wall
            if not self.can_move(direction, walls):
                continue

            # Calculate the score: distance to ghost from the next position
            next_center = pacman_pos + direction * TILE_SIZE
            # We want to MAXIMIZE the distance to the ghost
            score = next_center.distance_squared_to(closest_ghost_pos)

            if score > best_score:
                best_score = score
                best_direction = direction

        return best_direction


    def update(self, walls: List[pygame.Rect], pellets: List[Pellet], ghosts: List['Ghost'], dt: float) -> None:
        """MODIFIED: Handles mode switching (Seek vs. Flee) and applies AI logic."""
        
        # 1. Update cooldown timer
        if self.flee_cooldown > 0.0:
            self.flee_cooldown = max(0.0, self.flee_cooldown - dt)

        # Only make a new decision when centered on a tile
        if not self.at_tile_center():
            if not self.move(walls):
                self.direction.update(0, 0)
            return

        # --- Decision Time (At tile center) ---
        
        # Identify non-frightened ghosts for danger assessment
        danger_ghosts = [g for g in ghosts if not g.frightened]
        danger = self.is_in_danger(danger_ghosts)
        
        # 2. Mode Transition Logic
        if danger and self.mode != "flee":
            self.mode = "flee"
        elif not danger and self.mode == "flee"