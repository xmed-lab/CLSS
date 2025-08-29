import torch
import torch.linalg as LA

def compute_formula_torch(Lm):
    """
    使用 PyTorch 计算公式：
    min{ σ₁(Lm)/2, (λ₁² * σ₁(Lm) * min_{i,j,k} |α_{ik} - α_{jk}|) / (8√2 * λn * n * ||Lm||∞) }
    """
    n = 32
    
    # 确保 Lm 是浮点类型
    if Lm.dtype != torch.float32 and Lm.dtype != torch.float64:
        Lm = Lm.float()
    
    # 1. 计算最小奇异值 σ₁(Lm)
    sigma1 = torch.min(LA.svdvals(Lm))
    
    # 2. 计算特征值和特征向量
    eigenvalues, eigenvectors = LA.eigh(Lm)  # 使用 eigh 因为 Lm 是实对称矩阵

    # 按特征值非降序排列 (eigh 已经返回升序排列的特征值)
    lambda1 = eigenvalues[0]  # 最小特征值 λ₁
    lambda_n = eigenvalues[-1]  # 最大特征值 λn
    
    # 3. 计算 min_{i,j,k} |α_{ik} - α_{jk}|
    min_diff = torch.tensor(float('inf'))
    for i in range(n):
        for j in range(n):
            if i != j:  # 排除相同特征向量
                diff = torch.abs(eigenvectors[:, i] - eigenvectors[:, j])
                current_min = torch.min(diff)
                if current_min < min_diff:
                    min_diff = current_min
    
    # 4. 计算 ||Lm||∞（行范数）
    inf_norm = torch.max(torch.sum(torch.abs(Lm), dim=1))
    
    # 5. 计算两项结果
    term1 = sigma1 / 2
    numerator = lambda1**2 * sigma1 * min_diff
    denominator = 8 * torch.sqrt(torch.tensor(2.0)) * lambda_n * n * inf_norm
    term2 = numerator / denominator if denominator != 0 else torch.tensor(float('inf'))
    
    # 6. 取最小值
    result = torch.min(term1, term2)
    
    return result, term1, term2

# 示例使用
if __name__ == "__main__":
    # 创建一个示例矩阵（替换为实际数据）
    Lm = torch.tensor([[2.0, -1.0, 0.0],
                       [-1.0, 2.0, -1.0],
                       [0.0, -1.0, 2.0]])
    
    result, term1, term2 = compute_formula_torch(Lm)
    print(f"计算结果: {result.item()}")
    print(f"第一项值: {term1.item()}")
    print(f"第二项值: {term2.item()}")
    
    # 如果有 GPU，可以将计算迁移到 GPU 上
    if torch.cuda.is_available():
        device = torch.device("cuda")
        Lm_gpu = Lm.to(device)
        result_gpu, term1_gpu, term2_gpu = compute_formula_torch(Lm_gpu)
        print(f"GPU计算结果: {result_gpu.item()}")