import heapq

def a_star(s, g, graph, h):
    def neighbors(n):
        return graph.get(n, [])
    
    open_set = [(h[s], 0, s)]
    heapq.heapify(open_set)
    visit = {s: None}
    g_val = {s: 0}
    
    while open_set:
        _, cost, n = heapq.heappop(open_set)
        
        if n == g:
            path = []
            while n is not None:
                path.append(n)
                n = visit[n]
            return path[::-1]
        
        for m, w in neighbors(n):
            new_g = g_val[n] + w
            if m not in g_val or new_g < g_val[m]:
                visit[m] = n
                g_val[m] = new_g
                f = new_g + h[m]
                heapq.heappush(open_set, (f, new_g, m))
    
    return None


graph = {
    'S': [('A', 1), ('B', 4)],
    'A': [('B', 2), ('C', 5)],
    'B': [('C', 1)],
    'C': [('G', 3)],
}

heuristic = {
    'S': 7,
    'A': 4,
    'B': 2,
    'C': 1,
    'G': 0
}

print(a_star('S', 'G', graph, heuristic))