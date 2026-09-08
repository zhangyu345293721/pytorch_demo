import heapq

def manhattan(a, b):
    """曼哈顿距离 h(n)，4方向移动"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_steps(grid, start, goal):
    """
    A*，只返回最短路径步数，不返回路径
    Args:
        grid: 0空地，1障碍物
        start: (x,y)起点
        goal: (x,y)终点
    Returns:
        int: 最小步数；无路径返回None
    """
    rows = len(grid)
    cols = len(grid[0])

    open_heap = []
    heapq.heappush(open_heap, (0, start[0], start[1]))

    # g_cost：key坐标，value=起点到此点的步数(g代价)
    g_cost = {}
    g_cost[start] = 0

    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while open_heap:
        f_current, x, y = heapq.heappop(open_heap)
        current = (x, y)

        # 弹出终点，直接读取g_cost，就是步数！
        if current == goal:
            return g_cost[current]

        for dx, dy in dirs:
            nx = x + dx
            ny = y + dy
            if 0 <= nx < rows and 0 <= ny < cols and grid[nx][ny] == 0:
                tentative_g = g_cost[current] + 1

                neighbor = (nx, ny)
                if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                    g_cost[neighbor] = tentative_g
                    h = manhattan(neighbor, goal)
                    f = tentative_g + h
                    heapq.heappush(open_heap, (f, nx, ny))

    # 没有通路
    return None

if __name__ == "__main__":
    demo_grid = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ]
    start_pt = (0, 0)
    goal_pt = (4, 4)
    step_count = a_star_steps(demo_grid, start_pt, goal_pt)
    if step_count is not None:
        print(f"最小步数 = {step_count}")
    else:
        print("没有通路")
