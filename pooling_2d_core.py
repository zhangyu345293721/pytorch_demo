import numpy as np

def pool2d(input_matrix, kernel_size, stride, mode='max'):
    """
    2D 池化操作
    
    Parameters:
    - input_matrix (ndarray): 输入矩阵（2D 数组）。
    - kernel_size (int): 池化核的大小（假设为正方形核）。
    - stride (int): 步幅大小。
    - mode (str): 池化模式，'max' 表示最大池化，'avg' 表示平均池化。
    
    Returns:
    - output_matrix (ndarray): 池化后的矩阵。
    """
    input_h, input_w = input_matrix.shape
    output_h = (input_h - kernel_size) // stride + 1
    output_w = (input_w - kernel_size) // stride + 1

    output_matrix = np.zeros((output_h, output_w))
    
    for i in range(output_h):
        for j in range(output_w):
            start_i = i * stride
            start_j = j * stride
            end_i = start_i + kernel_size
            end_j = start_j + kernel_size

            # 获取池化区域
            patch = input_matrix[start_i:end_i, start_j:end_j]
            
            if mode == 'max':
                output_matrix[i, j] = np.max(patch)  # 最大池化
            elif mode == 'avg':
                output_matrix[i, j] = np.mean(patch)  # 平均池化
            else:
                raise ValueError("Mode must be 'max' or 'avg'")
    
    return output_matrix

# 示例
input_matrix = np.array([
    [1, 2, 3, 0],
    [4, 5, 6, 1],
    [7, 8, 9, 2],
    [3, 4, 5, 6]
])

kernel_size = 2
stride = 2

# 最大池化
max_pooled = pool2d(input_matrix, kernel_size, stride, mode='max')
print("Max Pooling Result:")
print(max_pooled)

# 平均池化
avg_pooled = pool2d(input_matrix, kernel_size, stride, mode='avg')
print("\nAverage Pooling Result:")
print(avg_pooled)
