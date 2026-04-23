import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import multiprocessing as mp
import math
import copy
import time
import os
import subprocess
import shutil

# ==========================================
# 1. GAME ENVIRONMENT (Connect 4)
# ==========================================
class Connect4:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.action_size = self.cols

    def get_initial_state(self):
        return np.zeros((self.rows, self.cols), dtype=np.int8)

    def get_next_state(self, state, action, player):
        next_state = np.copy(state)
        for r in range(self.rows - 1, -1, -1):
            if next_state[r, action] == 0:
                next_state[r, action] = player
                break
        return next_state

    def get_valid_moves(self, state):
        return (state[0] == 0).astype(np.uint8)

    def check_game_over(self, state, action, player):
        if action is None:
            return False, 0

        r = 0
        while r < self.rows and state[r, action] == 0:
            r += 1
        if r == self.rows:
            r = self.rows - 1
        elif state[r, action] != player:
            r += 1

        directions = [(0, 1), (1, 0), (-1, 1), (1, 1)]
        for dr, dc in directions:
            count = 1
            for direction in [1, -1]:
                step = 1
                while True:
                    nr, nc = r + direction * dr * step, action + direction * dc * step
                    if 0 <= nr < self.rows and 0 <= nc < self.cols and state[nr, nc] == player:
                        count += 1
                        step += 1
                    else:
                        break
            if count >= 4:
                return True, 1

        if np.sum(self.get_valid_moves(state)) == 0:
            return True, 0
        return False, 0

# ==========================================
# 2. NEURAL NETWORK (Global Receptance MLP)
# ==========================================
class ResidualLinearBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        residual = x
        x = F.relu(self.ln1(self.fc1(x)))
        x = self.ln2(self.fc2(x))
        x += residual
        return F.relu(x)

class AlphaZeroMLP(nn.Module):
    def __init__(self, action_size=7, hidden_dim=256, num_blocks=6):
        super().__init__()
        self.action_size = action_size
        self.input_size = 2 * 6 * 7

        self.input_layer = nn.Linear(self.input_size, hidden_dim)
        self.res_blocks = nn.ModuleList([ResidualLinearBlock(hidden_dim) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh()
        )

    def forward(self, state):
        x = state.view(-1, self.input_size)
        x = F.relu(self.input_layer(x))

        for block in self.res_blocks:
            x = block(x)

        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value

# ==========================================
# 3. STRICT ALPHAZERO MCTS
# ==========================================
class Node:
    def __init__(self, state, parent=None, action_taken=None, prior=0):
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, game, player, policy_probs, valid_moves):
        for action, prob in enumerate(policy_probs):
            if valid_moves[action] == 1:
                next_state = game.get_next_state(self.state, action, player)
                self.children[action] = Node(next_state, self, action, prior=prob)

class MCTS:
    def __init__(self, game, model, num_simulations=400, c_puct=1.25):
        self.game = game
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def search(self, initial_state, player, add_noise=True):
        root = Node(initial_state)

        encoded_state = torch.tensor(self.get_encoded_state(initial_state, player)).unsqueeze(0).to(self.device)
        logits, _ = self.model(encoded_state)
        logits = logits.cpu().numpy().flatten()

        valid_moves = self.game.get_valid_moves(initial_state)
        logits[valid_moves == 0] = -1e9

        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        policy_probs = exp_logits / np.sum(exp_logits)

        if add_noise:
            epsilon = 0.25
            alpha = 1.0
            noise = np.random.dirichlet([alpha] * np.sum(valid_moves))

            noise_idx = 0
            for a in range(self.game.action_size):
                if valid_moves[a]:
                    policy_probs[a] = (1 - epsilon) * policy_probs[a] + epsilon * noise[noise_idx]
                    noise_idx += 1

        root.expand(self.game, player, policy_probs, valid_moves)

        for _ in range(self.num_simulations):
            node = root
            search_player = player

            while len(node.children) > 0:
                best_u, best_child = -float('inf'), None
                for child in node.children.values():
                    u = child.q_value + self.c_puct * child.prior * (
                        math.sqrt(max(1, node.visit_count)) / (1 + child.visit_count)
                    )
                    if u > best_u:
                        best_u = u
                        best_child = child
                node = best_child
                search_player *= -1

            is_terminal, reward = self.game.check_game_over(node.state, node.action_taken, search_player * -1)

            if is_terminal:
                value = -reward
            else:
                encoded_state = torch.tensor(self.get_encoded_state(node.state, search_player)).unsqueeze(0).to(self.device)
                logits, value = self.model(encoded_state)
                logits = logits.cpu().numpy().flatten()
                value = value.item()

                valid_moves = self.game.get_valid_moves(node.state)
                logits[valid_moves == 0] = -1e9
                max_logit = np.max(logits)
                exp_logits = np.exp(logits - max_logit)
                policy_probs = exp_logits / np.sum(exp_logits)

                node.expand(self.game, search_player, policy_probs, valid_moves)

            value = -value

            while node is not None:
                node.visit_count += 1
                node.value_sum += value
                value = -value
                node = node.parent

        action_probs = np.zeros(self.game.action_size)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        return action_probs / np.sum(action_probs)

    def get_encoded_state(self, state, player):
        encoded = np.zeros((2, self.game.rows, self.game.cols), dtype=np.float32)
        encoded[0] = (state == player).astype(np.float32)
        encoded[1] = (state == -player).astype(np.float32)
        return encoded

# ==========================================
# 4. STRONGER ALPHA-BETA SOLVER
# ==========================================
class AlphaBetaBot:
    WIN_SCORE = 10_000_000
    COL_ORDER = (3, 2, 4, 1, 5, 0, 6)

    POSITION_WEIGHTS = np.array([
        [3, 4, 5, 7, 5, 4, 3],
        [4, 6, 8,10, 8, 6, 4],
        [5, 8,11,13,11, 8, 5],
        [5, 8,11,13,11, 8, 5],
        [4, 6, 8,10, 8, 6, 4],
        [3, 4, 5, 7, 5, 4, 3],
    ], dtype=np.int32)

    def __init__(self, game, depth=4):
        self.game = game
        self.depth = depth
        self.tt = {}
        self.windows = self._build_windows()

    def _build_windows(self):
        windows = []

        for r in range(self.game.rows):
            for c in range(self.game.cols - 3):
                windows.append([(r, c + i) for i in range(4)])

        for r in range(self.game.rows - 3):
            for c in range(self.game.cols):
                windows.append([(r + i, c) for i in range(4)])

        for r in range(self.game.rows - 3):
            for c in range(self.game.cols - 3):
                windows.append([(r + i, c + i) for i in range(4)])

        for r in range(3, self.game.rows):
            for c in range(self.game.cols - 3):
                windows.append([(r - i, c + i) for i in range(4)])

        return windows

    def _last_move_row(self, state, action):
        for r in range(self.game.rows):
            if state[r, action] != 0:
                return r
        return None

    def _has_won(self, state, action, player):
        if action is None:
            return False

        row = self._last_move_row(state, action)
        if row is None or state[row, action] != player:
            return False

        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                nr = row + sign * dr
                nc = action + sign * dc
                while (
                    0 <= nr < self.game.rows and
                    0 <= nc < self.game.cols and
                    state[nr, nc] == player
                ):
                    count += 1
                    nr += sign * dr
                    nc += sign * dc
            if count >= 4:
                return True
        return False

    def _terminal_score(self, state, last_action, last_player, ply):
        if last_action is not None and self._has_won(state, last_action, last_player):
            return True, -(self.WIN_SCORE - ply)

        if not np.any(self.game.get_valid_moves(state)):
            return True, 0

        return False, 0

    def _playable_mask(self, state):
        playable = np.zeros_like(state, dtype=np.bool_)
        for c in range(self.game.cols):
            for r in range(self.game.rows - 1, -1, -1):
                if state[r, c] == 0:
                    playable[r, c] = True
                    break
        return playable

    def _evaluate_window(self, values, coords, playable, player):
        opp = -player
        player_count = sum(v == player for v in values)
        opp_count = sum(v == opp for v in values)
        empty_count = 4 - player_count - opp_count

        if player_count > 0 and opp_count > 0:
            return 0

        playable_empties = sum(
            1 for (r, c), v in zip(coords, values) if v == 0 and playable[r, c]
        )

        score = 0

        if player_count == 4:
            return 100000
        if opp_count == 4:
            return -100000

        if player_count == 3 and empty_count == 1:
            score += 1200 if playable_empties == 1 else 120
        elif player_count == 2 and empty_count == 2:
            score += 40 + 10 * playable_empties
        elif player_count == 1 and empty_count == 3:
            score += 2

        if opp_count == 3 and empty_count == 1:
            score -= 1500 if playable_empties == 1 else 150
        elif opp_count == 2 and empty_count == 2:
            score -= 50 + 10 * playable_empties

        return score

    def _window_and_position_score(self, state, player):
        opp = -player
        score = int(
            np.sum(self.POSITION_WEIGHTS[state == player]) -
            np.sum(self.POSITION_WEIGHTS[state == opp])
        )

        playable = self._playable_mask(state)
        for coords in self.windows:
            values = [int(state[r, c]) for r, c in coords]
            score += self._evaluate_window(values, coords, playable, player)

        return int(score)

    def _immediate_winning_moves(self, state, player):
        valid_moves = self.game.get_valid_moves(state)
        wins = []
        for col in self.COL_ORDER:
            if valid_moves[col]:
                next_state = self.game.get_next_state(state, col, player)
                if self._has_won(next_state, col, player):
                    wins.append(col)
        return wins

    def evaluate_board(self, state, player):
        score = self._window_and_position_score(state, player)
        my_wins = len(self._immediate_winning_moves(state, player))
        opp_wins = len(self._immediate_winning_moves(state, -player))

        score += 6000 * my_wins
        score -= 7000 * opp_wins

        if my_wins >= 2:
            score += 15000
        if opp_wins >= 2:
            score -= 20000

        return int(score)

    def _ordered_moves(self, state, player, tt_move=None):
        valid_moves = self.game.get_valid_moves(state)
        opp = -player
        opp_wins_now = set(self._immediate_winning_moves(state, opp))

        scored_moves = []
        for col in self.COL_ORDER:
            if not valid_moves[col]:
                continue

            next_state = self.game.get_next_state(state, col, player)

            if self._has_won(next_state, col, player):
                priority = 1_000_000_000
            else:
                priority = self._window_and_position_score(next_state, player)
                if col in opp_wins_now:
                    priority += 50_000

                opp_reply_wins = len(self._immediate_winning_moves(next_state, opp))
                if opp_reply_wins:
                    priority -= 60_000 * opp_reply_wins

                priority -= abs(3 - col)

            if tt_move is not None and col == tt_move:
                priority += 100_000_000

            scored_moves.append((priority, col))

        scored_moves.sort(reverse=True)
        return [col for _, col in scored_moves]

    def negamax(self, state, depth, alpha, beta, player, last_action=None, ply=0):
        alpha_orig = alpha
        beta_orig = beta

        key = (state.tobytes(), player)
        entry = self.tt.get(key)
        tt_move = None

        if entry is not None:
            tt_move = entry["best_move"]
            if entry["depth"] >= depth:
                if entry["flag"] == "EXACT":
                    return entry["best_move"], entry["score"]
                elif entry["flag"] == "LOWER":
                    alpha = max(alpha, entry["score"])
                elif entry["flag"] == "UPPER":
                    beta = min(beta, entry["score"])

                if alpha >= beta:
                    return entry["best_move"], entry["score"]

        last_player = -player if last_action is not None else None
        is_terminal, terminal_score = self._terminal_score(state, last_action, last_player, ply)
        if is_terminal:
            return None, terminal_score

        if depth == 0:
            return None, self.evaluate_board(state, player)

        valid_moves = self.game.get_valid_moves(state)
        if not np.any(valid_moves):
            return None, 0

        moves = self._ordered_moves(state, player, tt_move=tt_move)
        best_move = moves[0]
        best_score = -math.inf

        for col in moves:
            next_state = self.game.get_next_state(state, col, player)
            _, child_score = self.negamax(
                next_state,
                depth - 1,
                -beta,
                -alpha,
                -player,
                last_action=col,
                ply=ply + 1,
            )
            score = -child_score

            if score > best_score:
                best_score = score
                best_move = col

            alpha = max(alpha, best_score)
            if alpha >= beta:
                break

        flag = "EXACT"
        if best_score <= alpha_orig:
            flag = "UPPER"
        elif best_score >= beta_orig:
            flag = "LOWER"

        self.tt[key] = {
            "depth": depth,
            "score": best_score,
            "flag": flag,
            "best_move": best_move,
        }

        return best_move, best_score

    def get_action(self, state, player):
        my_wins = self._immediate_winning_moves(state, player)
        if my_wins:
            return my_wins[0]

        opp_wins = self._immediate_winning_moves(state, -player)
        if len(opp_wins) == 1:
            return opp_wins[0]

        if len(self.tt) > 200000:
            self.tt.clear()

        col, _ = self.negamax(state, self.depth, -math.inf, math.inf, player)
        if col is not None:
            return col

        valid_moves = self.game.get_valid_moves(state)
        for c in self.COL_ORDER:
            if valid_moves[c]:
                return c
        return 0

# ==========================================
# 4B. PERFECT-PLAY / SOLVED BOT WRAPPER
# ==========================================
class PascalPonsPerfectBot:
    """
    Wraps an external Pascal Pons solver executable.

    The solver expects a move-history string like:
      445362...
    using columns 1..7, one line per position.

    In analyze mode (-a), it returns one score per column.
    We choose the best legal move (highest score).
    """
    def __init__(self, game, solver_path="./connect4_solver", book_path="7x6.book", fallback_depth=6):
        self.game = game
        self.solver_path = solver_path
        self.book_path = book_path
        self.fallback_bot = AlphaBetaBot(game, depth=fallback_depth)

    def _solver_exists(self):
        return os.path.isfile(self.solver_path) and os.access(self.solver_path, os.X_OK)

    def _history_to_solver_string(self, move_history):
        return "".join(str(a + 1) for a in move_history)

    def _query_scores(self, move_history):
        if not self._solver_exists():
            raise FileNotFoundError(
                f"Solver executable not found or not executable: {self.solver_path}"
            )

        seq = self._history_to_solver_string(move_history)
        cmd = [self.solver_path, "-a"]

        if self.book_path is not None and len(self.book_path) > 0:
            cmd.extend(["-b", self.book_path])

        result = subprocess.run(
            cmd,
            input=seq + "\n",
            text=True,
            capture_output=True,
            check=True
        )

        stdout_lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not stdout_lines:
            raise RuntimeError(f"Solver returned no parseable output.\nSTDERR:\n{result.stderr}")

        line = stdout_lines[0]
        parts = line.split()

        # Non-empty sequence: output format is "<seq> s0 s1 s2 s3 s4 s5 s6"
        # Empty sequence: often just "s0 s1 s2 s3 s4 s5 s6"
        if seq and parts and parts[0] == seq:
            parts = parts[1:]

        if len(parts) < self.game.cols:
            raise RuntimeError(
                f"Could not parse solver output: '{line}'\nSTDERR:\n{result.stderr}"
            )

        scores = [int(x) for x in parts[:self.game.cols]]
        return scores

    def get_action(self, state, player, move_history):
        valid_moves = self.game.get_valid_moves(state)

        try:
            scores = self._query_scores(move_history)
            best_col = None
            best_score = -10**18

            for col in range(self.game.cols):
                if valid_moves[col] and scores[col] > best_score:
                    best_score = scores[col]
                    best_col = col

            if best_col is None:
                raise RuntimeError("Solver produced no legal move.")
            return best_col

        except Exception as e:
            print(f"[PerfectBot warning] {e}")
            print("[PerfectBot warning] Falling back to strong alpha-beta.")
            return self.fallback_bot.get_action(state, player)

# ==========================================
# 5. PARALLEL SELF-PLAY WORKER
# ==========================================
def self_play_worker(model_state_dict, num_games, simulations):
    torch.set_num_threads(1)
    game = Connect4()
    model = AlphaZeroMLP()
    model.load_state_dict(model_state_dict)
    model.eval()

    mcts = MCTS(game, model, num_simulations=simulations)
    worker_data = []

    for _ in range(num_games):
        state = game.get_initial_state()
        player = 1
        episode_data = []
        move_count = 0

        while True:
            action_probs = mcts.search(state, player, add_noise=True)

            temp = 1.0 if move_count < 15 else 0.01
            action_probs_temp = action_probs ** (1 / temp)
            action_probs_temp /= np.sum(action_probs_temp)

            action = np.random.choice(game.action_size, p=action_probs_temp)
            encoded_state = mcts.get_encoded_state(state, player)
            episode_data.append((encoded_state, action_probs_temp, player))

            state = game.get_next_state(state, action, player)
            is_terminal, reward = game.check_game_over(state, action, player)

            if is_terminal:
                for hist_state, hist_probs, hist_player in episode_data:
                    actual_value = reward if hist_player == player else -reward
                    worker_data.append((hist_state, hist_probs, actual_value))
                break

            player *= -1
            move_count += 1

    return worker_data

# ==========================================
# 6. EVALUATION AND MATCH MODES
# ==========================================
def print_board(s):
    symbols = {0: '.', 1: 'X', -1: 'O'}
    print("\n  0 1 2 3 4 5 6")
    print("  -------------")
    for r in range(s.shape[0]):
        row = " |" + "".join([symbols[s[r, c]] + "|" for c in range(s.shape[1])])
        print(row)
    print("  -------------\n")

def ask_yes_no(prompt, default="y"):
    raw = input(prompt).strip().lower()
    if raw == "":
        raw = default.lower()
    return raw in ("y", "yes", "1", "true", "t")

def ask_az_player():
    az_first = ask_yes_no("Should AlphaZero play first as X? (y/n, default y): ", default="y")
    return 1 if az_first else -1

def evaluate_vs_random(model, game, num_games=20, mcts_simulations=50):
    model.eval()
    mcts = MCTS(game, model, num_simulations=mcts_simulations)
    wins, draws, losses = 0, 0, 0

    for i in range(num_games):
        state = game.get_initial_state()
        bot_player = 1 if i % 2 == 0 else -1
        current_player = 1

        while True:
            if current_player == bot_player:
                action_probs = mcts.search(state, current_player, add_noise=False)
                action = np.argmax(action_probs)
            else:
                valid_moves = game.get_valid_moves(state)
                valid_indices = np.where(valid_moves == 1)[0]
                action = np.random.choice(valid_indices)

            state = game.get_next_state(state, action, current_player)
            is_terminal, reward = game.check_game_over(state, action, current_player)

            if is_terminal:
                if reward == 1:
                    if current_player == bot_player:
                        wins += 1
                    else:
                        losses += 1
                else:
                    draws += 1
                break
            current_player *= -1

    win_rate = wins / num_games
    print(f"Eval vs Random | Wins: {wins} | Losses: {losses} | Draws: {draws} | Win Rate: {win_rate*100:.1f}%")
    return win_rate

def play_human(model_path="alphazero_connect4_mlp.pth", mcts_simulations=400, az_player=-1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = Connect4()
    model = AlphaZeroMLP().to(device)

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nLoaded model from {model_path}")
    else:
        print("\nNo trained model found. Playing against an untrained network.")

    model.eval()
    mcts = MCTS(game, model, num_simulations=mcts_simulations)

    state = game.get_initial_state()
    player = 1
    human_player = -az_player
    move_history = []

    print(f"AlphaZero is {'X (first)' if az_player == 1 else 'O (second)'}")
    print(f"Human is {'X (first)' if human_player == 1 else 'O (second)'}")

    while True:
        print_board(state)
        if player == human_player:
            valid_moves = game.get_valid_moves(state)
            while True:
                try:
                    action = int(input("Enter your move (0-6): "))
                    if 0 <= action <= 6 and valid_moves[action] == 1:
                        break
                    print("Invalid move. Try again.")
                except ValueError:
                    print("Please enter a valid integer.")
        else:
            print("AlphaZero is thinking...")
            action_probs = mcts.search(state, player, add_noise=False)
            action = np.argmax(action_probs)
            print(f"AlphaZero chooses column {action}")

        state = game.get_next_state(state, action, player)
        move_history.append(action)
        is_terminal, reward = game.check_game_over(state, action, player)

        if is_terminal:
            print_board(state)
            if reward == 1:
                if player == human_player:
                    print("Game Over! You win!")
                else:
                    print("Game Over! AlphaZero wins!")
            else:
                print("Game Over! It's a draw.")
            break

        player *= -1

def play_vs_alphabeta(model_path="alphazero_connect4_mlp.pth", mcts_simulations=400, ab_depth=4, az_player=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = Connect4()

    model = AlphaZeroMLP().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nLoaded AlphaZero model from {model_path}")
    else:
        print("\nNo trained model found. Playing untrained network vs Alpha-Beta.")
    model.eval()
    mcts = MCTS(game, model, num_simulations=mcts_simulations)

    ab_bot = AlphaBetaBot(game, depth=ab_depth)
    print(f"Initialized AlphaBeta Bot with search depth {ab_depth}")

    state = game.get_initial_state()
    player = 1
    ab_player = -az_player
    move_history = []

    print(f"AlphaZero is {'X (first)' if az_player == 1 else 'O (second)'}")
    print(f"Alpha-Beta is {'X (first)' if ab_player == 1 else 'O (second)'}")

    while True:
        print_board(state)
        if player == az_player:
            print("AlphaZero is thinking...")
            start = time.time()
            action_probs = mcts.search(state, player, add_noise=False)
            action = np.argmax(action_probs)
            print(f"AlphaZero chooses column {action} (Time: {time.time()-start:.2f}s)")
        else:
            print("Alpha-Beta is thinking...")
            start = time.time()
            action = ab_bot.get_action(state, player)
            print(f"Alpha-Beta chooses column {action} (Time: {time.time()-start:.2f}s)")

        state = game.get_next_state(state, action, player)
        move_history.append(action)
        is_terminal, reward = game.check_game_over(state, action, player)

        if is_terminal:
            print_board(state)
            if reward == 1:
                if player == az_player:
                    print("Game Over! AlphaZero wins!")
                else:
                    print("Game Over! Alpha-Beta wins!")
            else:
                print("Game Over! It's a draw.")
            break

        player *= -1

def play_vs_gto(
    model_path="alphazero_connect4_mlp.pth",
    mcts_simulations=400,
    az_player=-1,
    solver_path="./connect4_solver",
    book_path="7x6.book"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    game = Connect4()

    model = AlphaZeroMLP().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"\nLoaded AlphaZero model from {model_path}")
    else:
        print("\nNo trained model found. Playing untrained network vs PerfectBot.")
    model.eval()
    mcts = MCTS(game, model, num_simulations=mcts_simulations)

    gto_bot = PascalPonsPerfectBot(
        game,
        solver_path=solver_path,
        book_path=book_path,
        fallback_depth=6
    )

    state = game.get_initial_state()
    player = 1
    gto_player = -az_player
    move_history = []

    print(f"AlphaZero is {'X (first)' if az_player == 1 else 'O (second)'}")
    print(f"PerfectBot is {'X (first)' if gto_player == 1 else 'O (second)'}")
    print(f"Solver path: {solver_path}")
    print(f"Book path:   {book_path}")

    while True:
        print_board(state)

        if player == az_player:
            print("AlphaZero is thinking...")
            start = time.time()
            action_probs = mcts.search(state, player, add_noise=False)
            action = np.argmax(action_probs)
            print(f"AlphaZero chooses column {action} (Time: {time.time()-start:.2f}s)")
        else:
            print("PerfectBot is thinking...")
            start = time.time()
            action = gto_bot.get_action(state, player, move_history)
            print(f"PerfectBot chooses column {action} (Time: {time.time()-start:.2f}s)")

        state = game.get_next_state(state, action, player)
        move_history.append(action)
        is_terminal, reward = game.check_game_over(state, action, player)

        if is_terminal:
            print_board(state)
            if reward == 1:
                if player == az_player:
                    print("Game Over! AlphaZero wins!")
                else:
                    print("Game Over! PerfectBot wins!")
            else:
                print("Game Over! It's a draw.")
            break

        player *= -1

# ==========================================
# 7. MAIN TRAINING LOOP
# ==========================================
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_iterations = 100
    games_per_worker = 10
    num_workers = mp.cpu_count() - 1 or 1
    simulations_per_move = 150
    batch_size = 128

    print(f"Training on: {device} | CPU Workers: {num_workers}")

    game = Connect4()
    model = AlphaZeroMLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    replay_buffer = []

    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration + 1}/{num_iterations} ---")

        model.cpu()
        state_dict = copy.deepcopy(model.state_dict())
        model.to(device)

        start_time = time.time()
        with mp.Pool(num_workers) as pool:
            results = pool.starmap(
                self_play_worker,
                [(state_dict, games_per_worker, simulations_per_move) for _ in range(num_workers)]
            )

        for worker_data in results:
            replay_buffer.extend(worker_data)

        if len(replay_buffer) > 50000:
            replay_buffer = replay_buffer[-50000:]

        print(f"Self-play generated {sum([len(r) for r in results])} moves in {time.time() - start_time:.1f}s")

        model.train()
        np.random.shuffle(replay_buffer)

        total_loss = 0
        num_batches = 0

        for i in range(0, len(replay_buffer), batch_size):
            batch = replay_buffer[i:i+batch_size]
            if len(batch) < batch_size:
                continue

            states, target_policies, target_values = zip(*batch)
            states = torch.FloatTensor(np.array(states)).to(device)
            target_policies = torch.FloatTensor(np.array(target_policies)).to(device)
            target_values = torch.FloatTensor(np.array(target_values)).unsqueeze(1).to(device)

            optimizer.zero_grad()
            out_logits, out_values = model(states)

            log_policies = F.log_softmax(out_logits, dim=1)
            policy_loss = -torch.sum(target_policies * log_policies) / batch_size
            value_loss = F.mse_loss(out_values, target_values)

            loss = value_loss + policy_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        print(f"Loss: {total_loss/max(1, num_batches):.4f}")

        print("Evaluating current model vs Random...")
        win_rate = evaluate_vs_random(model, game, num_games=20, mcts_simulations=50)

        torch.save(model.state_dict(), "alphazero_connect4_mlp.pth")
        if win_rate >= 0.95:
            torch.save(model.state_dict(), f"alphazero_c4_iter_{iteration}_strong.pth")

    print("Training complete.")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    print("1. Train AlphaZero")
    print("2. Play Human vs AlphaZero")
    print("3. Play AlphaZero vs Alpha-Beta Solver")
    print("4. Play AlphaZero vs Perfect/Solved Bot")

    choice = input("Enter choice (1, 2, 3, or 4): ").strip()

    if choice == '1':
        train()

    elif choice == '2':
        az_player = ask_az_player()
        play_human(az_player=az_player)

    elif choice == '3':
        depth_str = input("Enter Alpha-Beta search depth (e.g., 4 or 5): ").strip()
        try:
            depth = int(depth_str)
        except ValueError:
            print("Invalid depth, defaulting to 4.")
            depth = 4

        az_player = ask_az_player()
        play_vs_alphabeta(ab_depth=depth, az_player=az_player)

    elif choice == '4':
        az_player = ask_az_player()
        solver_path = input("Enter path to solved solver executable [./connect4_solver]: ").strip()
        if solver_path == "":
            solver_path = "./connect4_solver"

        book_path = input("Enter path to opening book [7x6.book]: ").strip()
        if book_path == "":
            book_path = "7x6.book"

        play_vs_gto(
            az_player=az_player,
            solver_path=solver_path,
            book_path=book_path
        )

    else:
        print("Invalid choice.")
