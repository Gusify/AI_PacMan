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
        self.stuck_timer = 0.0
        self.stuck_threshold = 2.0  # seconds before considering stuck
        self.last_position = Vector2(start_pos)
        self.position_tolerance = TILE_SIZE  # how far can move before resetting timer
        self.search_range_multiplier = 1.0
        self.max_search_multiplier = 3.0

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

    def find_nearby_threats(self, ghosts: List['Ghost'], threat_distance: float = 100.0) -> List['Ghost']:
        """Find ghosts that are dangerously close and not frightened."""
        pacman_pos = Vector2(self.rect.center)
        threats = []
        
        for ghost in ghosts:
            if ghost.frightened:
                continue
            ghost_pos = Vector2(ghost.rect.center)
            distance = pacman_pos.distance_to(ghost_pos)
            if distance < threat_distance:
                threats.append(ghost)
        
        return threats

    def find_safest_direction(self, walls: List[pygame.Rect], threats: List['Ghost']) -> Vector2:
        """Find the direction that maximizes distance from threats."""
        if not threats:
            return Vector2(0, 0)
        
        best_direction = None
        best_score = -float('inf')
        
        for direction in CARDINAL_DIRECTIONS:
            dx = int(direction.x * TILE_SIZE)
            dy = int(direction.y * TILE_SIZE)
            candidate = self.rect.move(dx, dy)
            
            # Skip if this direction hits a wall
            if any(candidate.colliderect(wall) for wall in walls):
                continue
            
            # Calculate total distance from all threats
            future_pos = Vector2(candidate.center)
            total_distance = 0
            for threat in threats:
                threat_pos = Vector2(threat.rect.center)
                total_distance += future_pos.distance_to(threat_pos)
            
            if total_distance > best_score:
                best_score = total_distance
                best_direction = direction
        
        return best_direction if best_direction else Vector2(0, 0)

    def heuristic(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors(self, position: tuple[int, int], walls: List[pygame.Rect]) -> List[tuple[int, int]]:
        """Get all valid neighboring tile positions for a given position.

        Args:
            position: A tuple (x, y) representing a tile position in pixels
            walls: List of wall rectangles to check collisions against

        Returns:
            List of valid neighboring positions as (x, y) tuples
        """
        neighbors = []
        x, y = position

        for direction in CARDINAL_DIRECTIONS:
            # Calculate neighbor position (one tile away)
            neighbor_x = x + int(direction.x * TILE_SIZE)
            neighbor_y = y + int(direction.y * TILE_SIZE)

            # Create a rect at the neighbor position to check if it's valid
            neighbor_rect = pygame.Rect(neighbor_x, neighbor_y, TILE_SIZE, TILE_SIZE)

            # Check if this position doesn't collide with any walls
            if not any(neighbor_rect.colliderect(wall) for wall in walls):
                neighbors.append((neighbor_x, neighbor_y))

        return neighbors

    def astar(self, start, goal, walls: List[pygame.Rect]):
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
                    heapq.heappush(open_set, (f[neighbor], neighbor))

        return []

    def update(self, walls: List[pygame.Rect], pellets: List[Pellet], ghosts: List['Ghost'] = None) -> None:
        if ghosts is None:
            ghosts = []
        
        # Check for nearby threats
        threats = self.find_nearby_threats(ghosts, threat_distance=80.0)
        
        if self.at_tile_center():
            if threats:
                print("threats nearby")
                # RUN AWAY from ghosts!
                self.direction = self.find_safest_direction(walls, threats)
            else:
                # Find nearest pellet and calculate path to it
                nearest_pellet = self.find_nearest_pellet(pellets)
                if nearest_pellet:
                    start_pos = (self.rect.x, self.rect.y)
                    goal_pos = (
                        int(nearest_pellet.center.x - TILE_SIZE // 2),
                        int(nearest_pellet.center.y - TILE_SIZE // 2)
                    )
                    path = self.astar(start_pos, goal_pos, walls)

                    # Set direction based on first step in path
                    if path:
                        next_pos = path[0]
                        dx = next_pos[0] - start_pos[0]
                        dy = next_pos[1] - start_pos[1]

                        if dx > 0:
                            self.direction = Vector2(1, 0)
                        elif dx < 0:
                            self.direction = Vector2(-1, 0)
                        elif dy > 0:
                            self.direction = Vector2(0, 1)
                        elif dy < 0:
                            self.direction = Vector2(0, -1)

        # Move in the current direction
        if not self.move(walls):
            self.direction.update(0, 0)

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, self.color, self.rect.center, TILE_SIZE // 2 - 1)


class Ghost(Entity):
    def __init__(
        self,
        start_pos: Vector2,
        color: tuple[int, int, int],
        behavior: str,
        scatter_target: Vector2,
    ) -> None:
        super().__init__(start_pos, color, GHOST_SPEED)
        self.behavior = behavior
        self.scatter_target = Vector2(scatter_target)
        self.base_speed = GHOST_SPEED
        self.speed = self.base_speed
        self.frightened = False
        self.frightened_timer = 0.0

    def reset(self) -> None:
        super().reset()
        self.speed = self.base_speed
        self.frightened = False
        self.frightened_timer = 0.0

    def set_frightened(self, duration: float) -> None:
        self.frightened = True
        self.frightened_timer = duration
        self.speed = max(1, self.base_speed - 1)

    def update_state(self, dt: float) -> None:
        if not self.frightened:
            return
        self.frightened_timer = max(0.0, self.frightened_timer - dt)
        if self.frightened_timer == 0.0:
            self.frightened = False
            self.speed = self.base_speed

    def available_directions(self, walls: List[pygame.Rect]) -> List[Vector2]:
        options: List[Vector2] = []
        for direction in CARDINAL_DIRECTIONS:
            dx = int(direction.x * TILE_SIZE)
            dy = int(direction.y * TILE_SIZE)
            candidate = self.rect.move(dx, dy)
            if not any(candidate.colliderect(wall) for wall in walls):
                options.append(Vector2(direction))
        return options

    def choose_direction(
        self,
        walls: List[pygame.Rect],
        pacman: Pacman,
        allow_reverse: bool = False,
    ) -> Vector2:
        options = self.available_directions(walls)
        if not options:
            return Vector2(0, 0)

        if not allow_reverse and self.direction.length_squared():
            options = [opt for opt in options if opt != -self.direction]
            if not options:
                options = self.available_directions(walls)

        if self.frightened:
            return random.choice(options)

        if self.behavior == "chaser":
            target = Vector2(pacman.rect.center)
        elif self.behavior == "ambusher":
            ahead = Vector2(pacman.rect.center) + pacman.direction * TILE_SIZE * 4
            target = ahead
        else:
            target = Vector2(self.scatter_target)

        return min(options, key=lambda opt: self._distance_to_target(opt, target))

    def _distance_to_target(self, direction: Vector2, target: Vector2) -> float:
        next_center = Vector2(self.rect.center) + direction * TILE_SIZE
        return next_center.distance_squared_to(target)

    def update(self, walls: List[pygame.Rect], pacman: Pacman, dt: float) -> None:
        self.update_state(dt)
        if self.at_tile_center():
            self.direction = self.choose_direction(walls, pacman)

        if not self.move(walls):
            self.direction = self.choose_direction(walls, pacman, allow_reverse=True)
            self.move(walls)

    def draw(self, surface: pygame.Surface) -> None:
        color = FRIGHTENED_COLOR if self.frightened else self.color
        pygame.draw.circle(surface, color, self.rect.center, TILE_SIZE // 2 - 1)
        eye_color = BLACK
        pygame.draw.circle(surface, eye_color, (self.rect.centerx - 4, self.rect.centery - 2), 2)
        pygame.draw.circle(surface, eye_color, (self.rect.centerx + 4, self.rect.centery - 2), 2)


# --- Game loop -------------------------------------------------------------

class Game:
    def __init__(self) -> None:
        pygame.init()
        self.maze = Maze(LEVEL_LAYOUT)
        width = self.maze.pixel_width
        height = self.maze.pixel_height + HUD_HEIGHT
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Mini Pac-Man")
        self.clock = pygame.time.Clock()

        self.font = pygame.font.SysFont("arial", 24, bold=True)
        self.title_font = pygame.font.SysFont("arial", 32, bold=True)

        self.score = 0
        self.lives = 3
        self.power_timer = 0.0
        self.state = "ready"
        self.ready_timer = 2.0

        self.pacman = Pacman(self.maze.player_start)
        self.ghosts = self._create_ghosts()
        self.reset_game()

    def _create_ghosts(self) -> List[Ghost]:
        behaviors = ["chaser", "ambusher", "patrol", "patrol"]
        scatter_points = [
            Vector2(TILE_SIZE * 1.5, TILE_SIZE * 1.5),
            Vector2(self.maze.pixel_width - TILE_SIZE * 1.5, TILE_SIZE * 1.5),
            Vector2(TILE_SIZE * 1.5, self.maze.pixel_height - TILE_SIZE * 1.5),
            Vector2(
                self.maze.pixel_width - TILE_SIZE * 1.5,
                self.maze.pixel_height - TILE_SIZE * 1.5,
            ),
        ]

        ghosts: List[Ghost] = []
        for idx, start in enumerate(self.maze.ghost_starts):
            color = GHOST_COLORS[idx % len(GHOST_COLORS)]
            behavior = behaviors[idx % len(behaviors)]
            scatter_target = scatter_points[idx % len(scatter_points)]
            ghost = Ghost(start, color, behavior, scatter_target)
            ghosts.append(ghost)
        return ghosts

    def reset_game(self) -> None:
        self.score = 0
        self.lives = 3
        self.power_timer = 0.0
        self.state = "ready"
        self.ready_timer = 2.0
        self.pacman.reset()
        for ghost in self.ghosts:
            ghost.reset()
        self.pellets = self.maze.create_pellets()

    def reset_entities(self) -> None:
        self.pacman.reset()
        for ghost in self.ghosts:
            ghost.reset()
        self.power_timer = 0.0
        self.ready_timer = 1.5
        self.state = "life_lost"

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key in DIRECTION_VECTORS:
                    self.pacman.queue_direction(Vector2(DIRECTION_VECTORS[event.key]))
                if event.key == pygame.K_SPACE and self.state in {"win", "gameover"}:
                    self.reset_game()

    def update(self, dt: float) -> None:
        if self.state == "ready" or self.state == "life_lost":
            self.ready_timer = max(0.0, self.ready_timer - dt)
            if self.ready_timer == 0.0:
                self.state = "playing"
            return

        if self.state != "playing":
            return

        self.pacman.update(self.maze.walls, self.pellets, self.ghosts)
        for ghost in self.ghosts:
            ghost.update(self.maze.walls, self.pacman, dt)

        self._handle_pellet_collisions(dt)
        self._handle_ghost_collisions()

        if not self.pellets:
            self.state = "win"

    def _handle_pellet_collisions(self, dt: float) -> None:
        if self.power_timer > 0.0:
            self.power_timer = max(0.0, self.power_timer - dt)

        for pellet in self.pellets[:]:
            if pellet.collides(self.pacman.rect):
                self.pellets.remove(pellet)
                if pellet.power:
                    self.score += 50
                    self.power_timer = POWER_MODE_DURATION
                    for ghost in self.ghosts:
                        ghost.set_frightened(POWER_MODE_DURATION)
                else:
                    self.score += 10

    def _handle_ghost_collisions(self) -> None:
        for ghost in self.ghosts:
            if not self.pacman.rect.colliderect(ghost.rect):
                continue

            if ghost.frightened:
                self.score += 200
                ghost.reset()
            else:
                self.lives -= 1
                if self.lives <= 0:
                    self.state = "gameover"
                else:
                    self.reset_entities()
                break

    def draw(self) -> None:
        self.screen.fill(BLACK)
        self.maze.draw(self.screen)

        for pellet in self.pellets:
            pellet.draw(self.screen)

        self.pacman.draw(self.screen)
        for ghost in self.ghosts:
            ghost.draw(self.screen)

        self._draw_hud()

        if self.state == "ready":
            self._draw_center_text("Ready!", YELLOW)
        elif self.state == "life_lost":
            self._draw_center_text("Watch out!", WHITE)
        elif self.state == "win":
            self._draw_center_text("You Win! (SPACE to restart)", WHITE)
        elif self.state == "gameover":
            self._draw_center_text("Game Over (SPACE to restart)", WHITE)

        pygame.display.flip()

    def _draw_hud(self) -> None:
        hud_rect = pygame.Rect(0, self.maze.pixel_height, self.maze.pixel_width, HUD_HEIGHT)
        pygame.draw.rect(self.screen, (20, 20, 20), hud_rect)

        score_surface = self.font.render(f"Score: {self.score}", True, WHITE)
        lives_surface = self.font.render(f"Lives: {self.lives}", True, WHITE)

        self.screen.blit(score_surface, (10, self.maze.pixel_height + 10))
        self.screen.blit(
            lives_surface,
            (self.maze.pixel_width - lives_surface.get_width() - 10, self.maze.pixel_height + 10),
        )

        if self.power_timer > 0.0:
            timer_surface = self.font.render(f"Power: {self.power_timer:0.1f}s", True, POWER_COLOR)
            self.screen.blit(timer_surface, (10, self.maze.pixel_height + 30))

    def _draw_center_text(self, text: str, color: tuple[int, int, int]) -> None:
        label = self.title_font.render(text, True, color)
        rect = label.get_rect(center=(self.maze.pixel_width / 2, self.maze.pixel_height / 2))
        self.screen.blit(label, rect)

    def run(self) -> None:
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()


if __name__ == "__main__":
    Game().run()
