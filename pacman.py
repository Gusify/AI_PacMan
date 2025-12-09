import random
import sys
import heapq
from dataclasses import dataclass
from typing import List

import math
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
  "1000000000GGGGGG000000000001",
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


# --- Pac-Man with ghost-aware pathfinding ----------------------------------
class Pacman(Entity):
    def __init__(self, start_pos: Vector2) -> None:
        super().__init__(start_pos, YELLOW, PACMAN_SPEED)
        self.current_path = []
        self.target_pellet = None
        self.path_recalc_counter = 0
        self.mode = "pellet"  # "pellet" or "flee"
        self.simple_mode = False  # Fallback to simpler behavior when needed
        self.last_decision_time = 0
        self.decision_cooldown = 5  # Frames between major recalculations
        self.mode_switch_timer = 0
        self.flee_cooldown = 5 #tiles
        self.last_tile: tuple[int, int] | None = None # <--- ADD THIS LINE


    def find_nearest_pellet(self, pellets: List[Pellet], max_distance: float = None):
        """Fast nearest pellet search with optional distance limit"""
        if not pellets:
            return None
        
        pacman_pos = Vector2(self.rect.center)
        nearest = None
        min_dist_sq = float('inf')
        
        for pellet in pellets:
            dx = pellet.center.x - pacman_pos.x
            dy = pellet.center.y - pacman_pos.y
            dist_sq = dx*dx + dy*dy
            
            if max_distance and dist_sq > max_distance*max_distance:
                continue
                
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                nearest = pellet
        
        return nearest

    def fast_heuristic(self, pos1, pos2):
        """Faster Manhattan distance calculation"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_neighbors_fast(self, position: tuple[int, int], walls: List[pygame.Rect]) -> List[tuple[int, int]]:
        """Optimized neighbor check using pre-calculated wall grid if possible"""
        neighbors = []
        x, y = position
        
        # Try all four directions (optimized order for Pacman's common movement)
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx = x + dx * TILE_SIZE
            ny = y + dy * TILE_SIZE
            rect = pygame.Rect(nx, ny, TILE_SIZE, TILE_SIZE)
            
            # Early exit if collision found
            collision = False
            for wall in walls:
                if rect.colliderect(wall):
                    collision = True
                    break
            
            if not collision:
                neighbors.append((nx, ny))
        
        return neighbors

    def quick_danger_check(self, ghosts: List["Ghost"], danger_radius: float = TILE_SIZE * 6) -> bool:
        """Fast danger assessment without complex calculations"""
        pacman_pos = Vector2(self.rect.center)
        
        for ghost in ghosts:
            if not ghost.frightened:
                gx, gy = ghost.rect.center
                px, py = pacman_pos
                dx = abs(gx - px)
                dy = abs(gy - py)
                
                # Quick distance check (Manhattan-ish)
                if dx + dy < danger_radius * 1.5:
                    # More precise check only if close
                    if pacman_pos.distance_to(Vector2(gx, gy)) < danger_radius:
                        return True
        return False

    def get_immediate_move_options(self, walls: List[pygame.Rect], ghosts: List["Ghost"]):
        """Get safe move options from current position without full pathfinding"""
        current_tile = (self.rect.x, self.rect.y)
        options = []
        
        for neighbor in self.get_neighbors_fast(current_tile, walls):
            # Quick safety check for each neighbor
            tile_center = Vector2(neighbor[0] + TILE_SIZE//2, neighbor[1] + TILE_SIZE//2)
            safe = True
            
            for ghost in ghosts:
                if not ghost.frightened:
                    ghost_dist = tile_center.distance_to(Vector2(ghost.rect.center))
                    if ghost_dist < TILE_SIZE * 4:
                        safe = False
                        break
            
            if safe:
                options.append(neighbor)
        
        return options

    def simple_path_toward(self, target_pos: tuple, walls: List[pygame.Rect], max_steps: int = 20):
        """Simplified greedy pathfinding with step limit"""
        path = []
        current = (self.rect.x, self.rect.y)
        
        for _ in range(max_steps):
            if current == target_pos:
                break
                
            best_neighbor = None
            best_score = float('inf')
            
            for neighbor in self.get_neighbors_fast(current, walls):
                score = self.fast_heuristic(neighbor, target_pos)
                if score < best_score:
                    best_score = score
                    best_neighbor = neighbor
            
            if not best_neighbor or best_neighbor == current:
                break
                
            path.append(best_neighbor)
            current = best_neighbor
        
        return path

    def quick_escape_direction(self, ghosts: List["Ghost"], walls: List[pygame.Rect]):
        """Find the best immediate escape direction without full search"""
        if not self.at_tile_center():
            return None
            
        current_tile = (self.rect.x, self.rect.y)
        best_dir = None
        best_score = -float('inf')
        
        # Get only non-frightened ghosts
        threatening_ghosts = [g for g in ghosts if not g.frightened]
        
        # If no threatening ghosts, just pick most open direction
        if not threatening_ghosts:
            for neighbor in self.get_neighbors_fast(current_tile, walls):
                next_neighbors = self.get_neighbors_fast(neighbor, walls)
                score = len(next_neighbors) * 100
                if score > best_score:
                    best_score = score
                    best_dir = neighbor
            return best_dir
        
        for neighbor in self.get_neighbors_fast(current_tile, walls):
            tile_center = Vector2(neighbor[0] + TILE_SIZE//2, neighbor[1] + TILE_SIZE//2)
            
            # Find the CLOSEST threatening ghost to this tile
            closest_ghost = None
            closest_dist = float('inf')
            
            for ghost in threatening_ghosts:
                dist = tile_center.distance_to(Vector2(ghost.rect.center))
                if dist < closest_dist:
                    closest_dist = dist
                    closest_ghost = ghost
            
            # Start with base score
            score = 100
            
            # Score based on closest ghost distance
            if closest_dist < TILE_SIZE * 2:
                score -= 700  # Danger zone
            elif closest_dist < TILE_SIZE * 4:
                score -= 100  # Warning zone
            elif closest_dist < TILE_SIZE * 6:
                score -= 20   # Caution zone
            else:
                score += closest_dist / TILE_SIZE  # Safe zone bonus
            
            # Bonus/penalty for moving toward/away from closest ghost
            current_pos = Vector2(self.rect.center)
            if closest_ghost:
                dir_to_tile = (tile_center - current_pos).normalize()
                ghost_pos = Vector2(closest_ghost.rect.center)
                dir_to_ghost = (ghost_pos - current_pos).normalize()
                
                dot_product = dir_to_tile.dot(dir_to_ghost)
                if dot_product < -0.7:  # Moving directly away
                    score += 400
                elif dot_product < -0.3:  # Moving somewhat away
                    score += 200
                elif dot_product > 0.7:  # Moving directly toward
                    score -= 400
                elif dot_product > 0.3:  # Moving somewhat toward
                    score -= 200
            
            # Score based on openness
            next_neighbors = self.get_neighbors_fast(neighbor, walls)
            score += len(next_neighbors) * 120  # Strong weight for escape routes
            
            # Force move if it's the only option
            if len(self.get_neighbors_fast(current_tile, walls)) == 1:
                score += 5000  # Huge bonus to ensure movement
            
            # Avoid going back to previous position (if we have that info)
            if self.last_tile and neighbor == self.last_tile: # <--- MODIFIED
                score -= 400 # Strong penalty to force a turn or forward movement
            
            print(f"Tile {neighbor}: closest ghost dist={closest_dist/TILE_SIZE:.1f}tiles, score={score}")
            
            if score > best_score:
                best_score = score
                best_dir = neighbor
        
        print(f"Best escape dir: {best_dir}, Score: {best_score}")
        return best_dir

    def update(self, walls: List[pygame.Rect], pellets: List[Pellet], ghosts: List["Ghost"]) -> None:
        current_time = pygame.time.get_ticks()
        
        # Only do expensive calculations occasionally
        if current_time - self.last_decision_time < 200:  # ms
            self.simple_mode = True
        else:
            self.simple_mode = False
            self.last_decision_time = current_time
        
        # Danger checks for mode switching with hysteresis
        is_in_high_danger = self.quick_danger_check(ghosts, TILE_SIZE * 3)
        is_safe_to_exit_flee = not self.quick_danger_check(ghosts, TILE_SIZE * 7)

        if self.at_tile_center():
            # Mode switching (simplified)
            old_mode = self.mode
            if is_in_high_danger and self.mode != "flee":
                
                self.mode = "flee"
                self.mode_switch_timer = self.flee_cooldown  # Now this is in tiles
                print("switched to flee \n \n")
            elif is_safe_to_exit_flee and self.mode == "flee" and self.mode_switch_timer == 0:
                self.mode = "pellet"
                print("switched to pellet mode again \n\n")
                print("path: " + str(self.current_path))

            if self.mode_switch_timer > 0:
                self.mode_switch_timer -= 1

            if old_mode != self.mode:
                if old_mode == "pellet" and self.mode == "flee":
                    self.current_path = []  # Clear path on mode change
                    print("path cleared(old mode is not the same as the new mode)")
            
            # Decide what to do based on mode and whether we're in simple mode
            if self.simple_mode:
                # Use fast, simple behavior
                if self.mode == "flee":
                    # Just pick a safe direction
                    escape_dir = self.quick_escape_direction(ghosts, walls)
                    if escape_dir:
                        dx = escape_dir[0] - self.rect.x
                        dy = escape_dir[1] - self.rect.y
                        self.direction = Vector2(dx // TILE_SIZE if dx != 0 else 0,
                                                dy // TILE_SIZE if dy != 0 else 0)
                        # Don't set a path, just move
                        self.current_path = []
                else:
                    # Pellet mode - grab nearest pellet in simple mode
                    nearest = self.find_nearest_pellet(pellets, TILE_SIZE * 25)

                    if nearest:
                        self.target_pellet = nearest
                        # Try a very short path or direct movement
                        if len(self.current_path) <= 1:
                            target_tile = (int(nearest.center.x - TILE_SIZE//2),
                                         int(nearest.center.y - TILE_SIZE//2))
                            self.current_path = self.simple_path_toward(target_tile, walls, max_steps=5)
            
            else:
                # We have time for more complex calculations
                if self.mode == "flee":
                    # Find a safe spot to flee to, not a pellet
                    escape_dir = self.quick_escape_direction(ghosts, walls)
                    if escape_dir:
                        # Create a short path in the escape direction
                        path = [escape_dir]
                        start_pos = (self.rect.x, self.rect.y)

                        # Get direction vector (as tile size steps)
                        dir_x = (escape_dir[0] - start_pos[0])
                        dir_y = (escape_dir[1] - start_pos[1])

                        current_pos = escape_dir

                        for _ in range(2): # 2 more steps
                            next_pos = (current_pos[0] + dir_x, current_pos[1] + dir_y)

                            rect = pygame.Rect(next_pos[0], next_pos[1], TILE_SIZE, TILE_SIZE)
                            if any(rect.colliderect(wall) for wall in walls):
                                break

                            path.append(next_pos)
                            current_pos = next_pos

                        self.current_path = path
                        self.target_pellet = None # Explicitly clear pellet target
                    else:
                        # If no escape, maybe try to find a power pellet as a last resort
                        power_pellets = [p for p in pellets if p.power]
                        if power_pellets:
                            closest_power = self.find_nearest_pellet(power_pellets, TILE_SIZE * 30)
                            if closest_power:
                                self.target_pellet = closest_power
                                print(f"Fleeing to power pellet as last resort")
                        else:
                            # No safe pellets, just pick any
                            self.target_pellet = self.find_nearest_pellet(pellets, TILE_SIZE * 30)
                
                else:  # pellet mode
                    # Only recalc path when needed
                    if (not self.current_path or 
                        self.target_pellet not in pellets or
                        self.path_recalc_counter >= 15):
                        
                        # --- START MODIFICATION HERE ---
                        
                        # 1. Filter out all Power Pellets when in 'pellet' mode.
                        regular_pellets = [p for p in pellets if not p.power]
                        
                        
                        pellets_to_consider = regular_pellets if regular_pellets else pellets
                        # If there are no regular pellets left, he must eat the power pellet to clear the level.
                        
                        # Prioritize close pellets from the filtered list
                        close_pellets = [p for p in pellets_to_consider 
                                       if Vector2(self.rect.center).distance_to(p.center) < TILE_SIZE * 10]
                        
                        if close_pellets:
                            self.target_pellet = self.find_nearest_pellet(close_pellets)
                        else:
                            self.target_pellet = self.find_nearest_pellet(pellets_to_consider)
                        self.path_recalc_counter = 0
                    else:
                        self.path_recalc_counter += 1
                
                # Generate path to target (with step limit)
                if self.target_pellet:
                    start_pos = (self.rect.x, self.rect.y)
                    goal_pos = (int(self.target_pellet.center.x - TILE_SIZE//2),
                              int(self.target_pellet.center.y - TILE_SIZE//2))
                    
                    # Use simpler pathfinding with limit
                    self.current_path = self.simple_path_toward(goal_pos, walls, max_steps=80)
                    
                    if not self.current_path:
                        # Fallback: just move toward target
                        dx = goal_pos[0] - start_pos[0]
                        dy = goal_pos[1] - start_pos[1]
                        
                        if abs(dx) > abs(dy):
                            self.direction = Vector2(1 if dx > 0 else -1, 0)
                        else:
                            self.direction = Vector2(0, 1 if dy > 0 else -1)
            
            # Follow path if we have one
            if self.current_path:
                next_pos = self.current_path[0]
                if next_pos == (self.rect.x, self.rect.y):
                    self.current_path.pop(0)
                    if not self.current_path:
                        return
                
                next_pos = self.current_path[0]
                dx = next_pos[0] - self.rect.x
                dy = next_pos[1] - self.rect.y
                
                if dx != 0:
                    self.direction = Vector2(1 if dx > 0 else -1, 0)
                elif dy != 0:
                    self.direction = Vector2(0, 1 if dy > 0 else -1)
                
                # Remove reached waypoints
                if self.rect.x == next_pos[0] and self.rect.y == next_pos[1]:
                    self.current_path.pop(0)
        
        # Always try to move
        
        current_tile_for_move = (self.rect.x, self.rect.y)
        
        # Always try to move
        if not self.move(walls):
            # If stuck, clear path and try a random safe direction
            self.current_path = []
            print('path cleared - stuck')
            
            # Use logic that avoids the last tile, if known
            options = [
                opt for opt in self.get_neighbors_fast((self.rect.x, self.rect.y), walls)
                if opt != self.last_tile
            ]
            
            if not options:
                # If the only option is the last tile, we have to take it
                options = self.get_neighbors_fast((self.rect.x, self.rect.y), walls)
            
            if options:
                import random
                next_pos = random.choice(options)
                dx = next_pos[0] - self.rect.x
                dy = next_pos[1] - self.rect.y
                self.direction = Vector2(dx // TILE_SIZE if dx != 0 else 0,
                                       dy // TILE_SIZE if dy != 0 else 0)
        
        else: # Successful move happened
            # 1. Check if Pac-Man is now on a new tile (not just moving within a tile)
            is_new_tile = (self.rect.x, self.rect.y) != current_tile_for_move
            
            if is_new_tile:
                # 2. Update memory with the tile we just left
                self.last_tile = current_tile_for_move
    
    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.circle(surface, self.color, self.rect.center, TILE_SIZE // 2 - 1)
# --- Ghost class -----------------------------------------------------------

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
        self.last_tile = None # <--- ADD THIS LINE
        

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
        for idx, start in enumerate(self.maze.ghost_starts[:4]):
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
                    self.pacman.direction = Vector2(DIRECTION_VECTORS[event.key])
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
