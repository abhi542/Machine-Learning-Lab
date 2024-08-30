def ao_star(graph, heuristics, start):
    status = {}
    parents = {}
    solution = {}

    def neighbors(node):
        return graph.get(node, [])

    def get(node):
        return status.get(node, 0)

    def set(node, value):
        status[node] = value

    def get_heuristic(node):
        return heuristics.get(node, 0)

    def min_cost(node):
        min_cost = float('inf')
        min_nodes = []
        for child_set in neighbors(node):
            cost = sum(get_heuristic(child) + weight for child, weight in child_set)
            nodes = [child for child, _ in child_set]
            if cost < min_cost:
                min_cost = cost
                min_nodes = nodes
        return min_cost, min_nodes

    def AOStar(node, backtrack):
        if get(node) >= 0:  
            cost, children = min_cost(node)
            heuristics[node] = cost
            set(node, len(children))
            if all(get(child) == -1 for child in children):
                set(node, -1)
                solution[node] = children

            if node != start: 
                AOStar(parents[node], True)

            if not backtrack: 
                for child in children:
                    if get(child) == 0:  
                        parents[child] = node
                        AOStar(child, False)

    AOStar(start, False)
    print("Solution:", solution)


heuristics = {'A': 10, 'B': 6, 'C': 4, 'D': 2, 'E': 0}
graph = {
    'A': [[('B', 1)], [('C', 1)]],
    'B': [[('D', 1)]],
    'C': [[('D', 1), ('E', 1)]],
    'D': [[('E', 1)]]
}

ao_star(graph, heuristics, 'A')