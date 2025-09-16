from collections import deque
import heapq

class PipeNetwork:
    def __init__(self, total_junctions):
        self.graph = {i: [] for i in range(total_junctions)}

    def add_pipe(self, u, v, cost):
        self.graph[u].append((v, cost))
        self.graph[v].append((u, cost)) 

def bfs(network, start, goal):
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        current, path = queue.popleft()
        if current == goal:
            return path

        if current not in visited:
            visited.add(current)
            for neighbor, _ in network.graph[current]:
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))
    return None

def dfs(network, current, goal, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(current)
    path.append(current)

    if current == goal:
        return path

    for neighbor, _ in network.graph[current]:
        if neighbor not in visited:
            result = dfs(network, neighbor, goal, visited, path)
            if result:
                return result

    path.pop()
    return None

def depth_limited_search(network, current, goal, limit, visited=None, path=None):
    if visited is None:
        visited = set()
    if path is None:
        path = []

    visited.add(current)
    path.append(current)

    if current == goal:
        return path
    if limit == 0:
        visited.remove(current)
        path.pop()
        return None

    for neighbor, _ in network.graph[current]:
        if neighbor not in visited:
            result = depth_limited_search(network, neighbor, goal, limit - 1, visited, path)
            if result:
                return result

    visited.remove(current)
    path.pop()
    return None

def iterative_deepening_search(network, start, goal, max_depth):
    for depth in range(max_depth + 1):
        result = depth_limited_search(network, start, goal, depth)
        if result:
            return result
    return None

def uniform_cost_search(network, start, goal):
    heap = [(0, start, [start])] 
    visited = {}

    while heap:
        cost, current, path = heapq.heappop(heap)

        if current == goal:
            return path, cost

        if current in visited and visited[current] <= cost:
            continue

        visited[current] = cost

        for neighbor, edge_cost in network.graph[current]:
            total_cost = cost + edge_cost
            heapq.heappush(heap, (total_cost, neighbor, path + [neighbor]))

    return None, float('inf')

def main():
    n, m = map(int, input("Enter number of junctions and pipes: ").split())
    network = PipeNetwork(n)

    print("Enter pipe connections (u v cost):")
    for _ in range(m):
        u, v, cost = map(int, input().split())
        network.add_pipe(u, v, cost)

    start = int(input("\nEnter start junction: "))
    goal = int(input("Enter goal junction: "))

    print("\nBFS (ignores cost):", bfs(network, start, goal))
    print("DFS (ignores cost):", dfs(network, start, goal))

    limit = int(input("\nEnter depth limit for Depth-Limited Search: "))
    print("Depth-Limited Search:", depth_limited_search(network, start, goal, limit))

    max_depth = int(input("\nEnter max depth for Iterative Deepening Search: "))
    print("Iterative Deepening Search:", iterative_deepening_search(network, start, goal, max_depth))

    path, total_cost = uniform_cost_search(network, start, goal)
    print(f"\nUniform Cost Search: {path} with total cost: {total_cost}")

if __name__ == "__main__":
    main()
