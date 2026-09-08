import java.util.PriorityQueue;

public class DijkstraGrid2D {

    /**
     * 二维栅格Dijkstra
     * @param grid 地图 0通行，1障碍物
     * @param startX 起点x
     * @param startY 起点y
     * @param goalX 终点x
     * @param goalY 终点y
     * @return 最短步数；不可达返回-1
     */
    public static int dijkstra(int[][] grid, int startX, int startY, int goalX, int goalY) {
        int rows = grid.length;
        if (rows == 0) return -1;
        int cols = grid[0].length;

        // dist[x][y] 起点到(x,y)最短距离
        int[][] dist = new int[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                dist[i][j] = Integer.MAX_VALUE;
            }
        }

        // 最小堆 int[]{distance, x, y}
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> Integer.compare(a[0], b[0]));

        dist[startX][startY] = 0;
        pq.add(new int[]{0, startX, startY});

        // 4方向 上下左右
        int[][] dirs = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

        while (!pq.isEmpty()) {
            int[] curr = pq.poll();
            int d = curr[0];
            int x = curr[1];
            int y = curr[2];

            // 到达终点，直接返回步数
            if (x == goalX && y == goalY) {
                return d;
            }

            // 过期无效堆节点，跳过
            if (d > dist[x][y]) {
                continue;
            }

            for (int[] dir : dirs) {
                int nx = x + dir[0];
                int ny = y + dir[1];
                // 边界校验 + 障碍物判断
                if (nx >= 0 && nx < rows && ny >= 0 && ny < cols && grid[nx][ny] == 0) {
                    // 防止Integer.MAX_VALUE溢出
                    if (dist[x][y] != Integer.MAX_VALUE) {
                        int newDist = dist[x][y] + 1;
                        if (newDist < dist[nx][ny]) {
                            dist[nx][ny] = newDist;
                            pq.add(new int[]{newDist, nx, ny});
                        }
                    }
                }
            }
        }
        // 队列空，没有通路
        return -1;
    }

    public static void main(String[] args) {
        int[][] demoGrid = {
                {0, 0, 0, 0, 0},
                {0, 1, 1, 1, 0},
                {0, 0, 0, 0, 0},
                {1, 0, 1, 0, 0},
                {0, 0, 0, 0, 0}
        };
        int steps = dijkstra(demoGrid, 0, 0, 4, 4);
        if (steps != -1) {
            System.out.println("Dijkstra最短步数 = " + steps);
        } else {
            System.out.println("无可行路径");
        }
    }
}
