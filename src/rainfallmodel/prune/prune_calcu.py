import torch
import torch.nn.utils.prune as prune
import torch.nn as nn


def do_prune_calcu(row_value:str, col_value:str, scale_value:str, prune_method:str, prune_amount:str) -> str:
    """
    演示剪枝计算逻辑
    """

    class MatrixHolder(torch.nn.Module):
        def __init__(self, matrix):
            super().__init__()
            self.weight = nn.Parameter(matrix)

    result = ""
    original_data = torch.randn(int(row_value), int(col_value)) * float(scale_value)
    holder = MatrixHolder(original_data)
    result += "\n原始数据:" +  str(original_data) + "  "
    result += "\n   "
    result += "\n   "
    

    amount = float(prune_amount)
    if "RandomStructured" == prune_method:
        prune.random_unstructured(holder, name="weight", amount=amount)
    elif "L1Unstructured" == prune_method:
        prune.l1_unstructured(holder, name="weight", amount=amount)
    
    result += "\n剪枝后的数据:" +  str(holder.weight) + "  "
    return result
