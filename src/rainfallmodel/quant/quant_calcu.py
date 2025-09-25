import torch
from bitsandbytes import functional as F
import html


def do_quant_calcu(row_value:str, col_value:str, scale_value:str) -> list:
    """
    基本的量化和反量化演示，便于理解
    """
    
    result = ""
    original_data = torch.randn(int(row_value), int(col_value)) * float(scale_value)
    result += "\n原始数据shape:" + str(original_data.shape) + "  "
    result += "\n原始数据dtype:" + str(original_data.dtype) + "  "
    result += "\n原始数据范围:" +  str(original_data.min()) + " ~ " + str(original_data.max()) + "  "
    result += "\n原始数据:" +  str(original_data) + "  "
    result += "\n   "
    result += "\n   "
    

    quantized_data, quant_state = F.quantize_blockwise(original_data)
    result += "\n量化后数据shape:" + str(quantized_data.shape) + "  "
    result += "\n量化后数据dtype:" + str(quantized_data.dtype) + "  "
    result += "\n量化后数据:" + str(quantized_data) + "  "
    result += "\n   "
    result += "\n   "
    

    dequantized_data = F.dequantize_blockwise(quantized_data, quant_state)
    result += "\n反量化后数据shape:" + str(dequantized_data.shape) + "  "
    result += "\n反量化后数据dtype:" + str(dequantized_data.dtype) + "  "
    result += f"\n反量化后数据: {dequantized_data}  "
    result += "\n   "
    result += "\n   "
    
    # 计算量化误差
    error = torch.abs(original_data - dequantized_data)
    result += "\n最大误差:" + str(error.max()) + "  "
    result += "\n平均误差:" + str(error.mean()) + "  "
    result += f"\n相对误差: {(error.mean() / original_data.abs().mean() * 100):.4f}%"
    
    return html.escape(result)




