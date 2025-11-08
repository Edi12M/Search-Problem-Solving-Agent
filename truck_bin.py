import heapq            
import random          
from collections import defaultdict  
import math            
import pandas as pd     
import matplotlib.pyplot as plt 

# ---------------------------------------------------------------------------
# Section 1: Graph Representation
# ---------------------------------------------------------------------------

class CityGraph:
    """Undirected weighted graph for city roads."""
    def __init__(self):
        # Dictionary: node -> list of tuples (neighbor_node, travel_cost)
        self.edges = defaultdict(list)

    def add_edge(self, u, v, cost):
        """Adds bidirectional edge with cost."""
        self.edges[u].append((v, cost))  # u -> v
        self.edges[v].append((u, cost))  # v -> u (undirected)

    def neighbors(self, node):
        """Returns list of (neighbor, cost) pairs for a given node."""
        return self.edges[node]


# ---------------------------------------------------------------------------
# Section 2: Waste Collection State and Problem Definition
# ---------------------------------------------------------------------------

class WasteState:
    """Represents a search state: (current position, collected bins, remaining capacity)."""
    def __init__(self, current, collected, capacity_left, cost=0):
        self.current = current                  # Current location of the truck
        self.collected = frozenset(collected)   # Immutable set of collected bins
        self.capacity_left = capacity_left      # Remaining truck capacity
        self.cost = cost                        # Cumulative cost to reach this state

    def __lt__(self, other):
        # Allows states to be compared in heapq based on cost
        return self.cost < other.cost

    def __hash__(self):
        # Hashable for use in sets/dictionaries
        return hash((self.current, self.collected, self.capacity_left))

    def __eq__(self, other):
        # Equality check to detect duplicate states
        return (self.current == other.current and
                self.collected == other.collected and
                self.capacity_left == other.capacity_left)


class WasteCollectionProblem:
    """Defines the waste collection search domain."""
    def __init__(self, graph, bins, start='Depot', unload='Depot', capacity=3):
        self.graph = graph       # CityGraph object
        self.bins = bins         # Dict: node -> waste amount
        self.start = start       # Starting node (usually depot)
        self.unload = unload     # Node where truck unloads collected waste
        self.capacity = capacity # Truck capacity

    def initial_state(self):
        """Returns initial search state at depot with full capacity."""
        return WasteState(self.start, frozenset(), self.capacity, 0)

    def is_goal(self, state):
        """Goal state is when all bins have been collected."""
        return len(state.collected) == len(self.bins)

    def successors(self, state):
        """
        Generates all possible next states:
        - Move to neighboring nodes
        - Collect waste at current node (if any and capacity allows)
        - Unload at depot if capacity is not full
        """
        # Move to neighbors
        for neighbor, cost in self.graph.neighbors(state.current):
            new_cost = state.cost + cost
            yield (f"Move({neighbor})", WasteState(neighbor, state.collected, state.capacity_left, new_cost), cost)

        # Collect waste at current node
        if state.current in self.bins and state.current not in state.collected:
            waste_amt = self.bins[state.current]
            if state.capacity_left >= waste_amt:
                new_collected = set(state.collected) | {state.current}
                yield (f"Collect({state.current})",
                       WasteState(state.current, new_collected, state.capacity_left - waste_amt, state.cost), 0)

        # Unload waste at depot if truck is not full
        if state.current == self.unload and state.capacity_left < self.capacity:
            yield ("Unload", WasteState(state.current, state.collected, self.capacity, state.cost), 0)


# ---------------------------------------------------------------------------
# Section 3: Search Algorithms (UCS and A*)
# ---------------------------------------------------------------------------

def uniform_cost_search(problem):
    """Uniform Cost Search implementation."""
    frontier = []
    heapq.heappush(frontier, (0, problem.initial_state(), []))  # (priority, state, path)
    explored = set()  # Track visited states to avoid revisiting
    expansions = 0    # Count number of node expansions

    while frontier:
        cost, state, path = heapq.heappop(frontier)
        if state in explored:
            continue
        explored.add(state)
        expansions += 1

        if problem.is_goal(state):
            return path, cost, expansions

        for action, next_state, step_cost in problem.successors(state):
            heapq.heappush(frontier, (next_state.cost, next_state, path + [action]))

    return None, math.inf, expansions  # No solution found


def heuristic_basic(state, problem):
    """Basic distance-based heuristic (uninformed average edge heuristic)."""
    remaining = [n for n in problem.bins if n not in state.collected]
    if not remaining:
        return 0
    # Average cost of all edges as rough estimate
    avg_cost = sum(c for v in problem.graph.edges.values() for _, c in v) / max(1, len(problem.graph.edges))
    return avg_cost * len(remaining)  # Multiply by number of remaining bins


def heuristic_capacity(state, problem):
    """Admissible heuristic that includes unload estimation."""
    base = heuristic_basic(state, problem)
    # Total waste left to collect
    total_waste_left = sum(problem.bins[n] for n in problem.bins if n not in state.collected)
    # Estimate number of unloads required
    required_unloads = max(0, math.ceil(total_waste_left / problem.capacity) - 1)
    return base + required_unloads * 2  # Penalize additional unload trips


def a_star_search(problem, heuristic_fn):
    """Generic A* implementation using a given heuristic function."""
    frontier = []
    start = problem.initial_state()
    heapq.heappush(frontier, (heuristic_fn(start, problem), start, []))
    explored = set()
    expansions = 0

    while frontier:
        est_total, state, path = heapq.heappop(frontier)
        if state in explored:
            continue
        explored.add(state)
        expansions += 1

        if problem.is_goal(state):
            return path, state.cost, expansions

        for action, next_state, step_cost in problem.successors(state):
            g = next_state.cost           # Actual cost to reach next_state
            h = heuristic_fn(next_state, problem)  # Heuristic estimate
            heapq.heappush(frontier, (g + h, next_state, path + [action]))

    return None, math.inf, expansions


# ---------------------------------------------------------------------------
# Section 4: Random Scenario Generation and Visualization
# ---------------------------------------------------------------------------

def generate_random_graph(num_nodes=7):
    """Generates random connected graph with random edge costs."""
    g = CityGraph()
    nodes = [f"N{i}" for i in range(num_nodes)]
    # Create linear path to ensure connectivity
    for i in range(num_nodes - 1):
        cost = random.randint(2, 10)
        g.add_edge(nodes[i], nodes[i + 1], cost)
    # Add some extra random edges to increase connectivity
    for _ in range(num_nodes // 2):
        u, v = random.sample(nodes, 2)
        cost = random.randint(3, 12)
        g.add_edge(u, v, cost)
    return g, nodes


def generate_scenario(seed=0):
    """
    Generates a random waste collection scenario:
    - Random graph
    - Random bins and waste amounts
    - Random truck capacity
    """
    random.seed(seed)
    g, nodes = generate_random_graph(random.randint(6, 8))
    bins = {n: random.randint(1, 3) for n in random.sample(nodes[1:], random.randint(2, 4))}
    return WasteCollectionProblem(g, bins, start=nodes[0], unload=nodes[0], capacity=random.randint(3, 4))


def run_and_visualize(scenarios=3):
    """
    Runs experiments for multiple scenarios:
    - Runs UCS, A* with basic heuristic, and A* with capacity-aware heuristic
    - Prints results in a DataFrame
    - Plots total cost and expansions comparison charts
    """
    results = []

    for i in range(scenarios):
        problem = generate_scenario(i)
        for algo_name, algo_fn in [
            ("UCS", lambda p: uniform_cost_search(p)),
            ("A*_Basic", lambda p: a_star_search(p, heuristic_basic)),
            ("A*_Capacity", lambda p: a_star_search(p, heuristic_capacity)),
        ]:
            path, cost, expansions = algo_fn(problem)
            results.append({
                "Scenario": i,
                "Algorithm": algo_name,
                "Total Cost": cost,
                "Expansions": expansions,
                "Action Trace": " -> ".join(path or [])
            })
            print(f"[Scenario {i}] {algo_name}: cost={cost}, expansions={expansions}")

    # Convert results to pandas DataFrame
    df = pd.DataFrame(results)
    print("\n=== Waste Collection Search Results ===")
    print(df.to_string(index=False))

    # ---- Plot 1: Total Cost per Algorithm ----
    plt.figure(figsize=(10, 5))
    for algo in df["Algorithm"].unique():
        subset = df[df["Algorithm"] == algo]
        plt.plot(subset["Scenario"], subset["Total Cost"], marker='o', label=algo)
    plt.title("Total Cost Comparison Across Scenarios")
    plt.xlabel("Scenario")
    plt.ylabel("Total Cost")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- Plot 2: Node Expansions per Algorithm ----
    plt.figure(figsize=(10, 5))
    for algo in df["Algorithm"].unique():
        subset = df[df["Algorithm"] == algo]
        plt.plot(subset["Scenario"], subset["Expansions"], marker='s', label=algo)
    plt.title("Search Expansions Comparison Across Scenarios")
    plt.xlabel("Scenario")
    plt.ylabel("Number of Expansions")
    plt.legend()
    plt.grid(True)
    plt.show()

    return df


# ---------------------------------------------------------------------------
# Run Example Visualization
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run 5 random scenarios and visualize the results
    run_and_visualize(scenarios=5)
