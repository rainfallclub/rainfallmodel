import torch
from transformers import AutoModel, AutoTokenizer
import torch.nn.utils.prune as prune
from ..model.model_manager import get_real_model_path



def comma_string_to_list(input_string:str) -> list:
        """
        将逗号分隔的字符串转换为列表
        """
        items = input_string.split(",")
        items = [item.strip() for item in items]
        items = [item for item in items if item]
        return items


def get_prune_method(prune_method:str):
    if "L1Unstructured" == prune_method:
        return prune.L1Unstructured
    elif "RandomStructured" == prune_method:
        return prune.RandomStructured
    else:
        return None



def do_unstructure_global_prune(prune_conf:dict) -> None:
    """
    非结构化全局剪枝
    """
    model_path = prune_conf['unstructure_global_model_path']
    real_model_path = get_real_model_path(model_path)
    model = AutoModel.from_pretrained(real_model_path)
    tokenizer = AutoTokenizer.from_pretrained(real_model_path)
    
    print(f"剪枝前总参数: { sum(p.numel() for p in model.parameters())}")
    print(f"剪枝前非零参数: {sum(torch.count_nonzero(p).item() for p in model.parameters())}")

    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))   
    
    prune_rate = float(prune_conf['unstructure_global_prune_rate'])
    pruning_method = get_prune_method(prune_conf['unstructure_global_prune_method'])
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=pruning_method,
        amount=prune_rate,
    )

    for module, name in parameters_to_prune:
        prune.remove(module, name)
    print(f"剪枝后非零参数: {sum(torch.count_nonzero(p).item() for p in model.parameters())}")

    output_dir = prune_conf['unstructure_global_output_dir']
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("剪枝完成！")



def do_unstructure_local_prune(prune_conf:dict) -> None:
    """
    非结构化局部剪枝
    """
    model_path = prune_conf['unstructure_local_model_path']
    real_model_path = get_real_model_path(model_path)
    model = AutoModel.from_pretrained(real_model_path)
    tokenizer = AutoTokenizer.from_pretrained(real_model_path)
    
    do_prune_layers = int(prune_conf['unstructure_local_prune_layers'])
    prune_method = prune_conf['unstructure_local_prune_method']
    for layer_num in range(do_prune_layers):
       
        self_attn_flag = bool(prune_conf['unstructure_local_prune_self_attn_flag'])
        if self_attn_flag:
            self_attn_module_str = prune_conf['unstructure_local_prune_self_attn_modules']
            self_attn_list = comma_string_to_list(self_attn_module_str)
            self_attn_rate = float(prune_conf['unstructure_local_prune_self_attn_rate'])
            for proj in self_attn_list:
                module = getattr(model.layers[layer_num].self_attn, proj)
                print(f"剪枝前model.layers{layer_num}.self_attn.{proj}权重中非零参数的数量: {torch.sum(module.weight != 0)}")
                if prune_method == "L1Unstructured":
                    prune.l1_unstructured(module, name="weight", amount=self_attn_rate)
                elif prune_method == "RandomStructured":
                    prune.random_unstructured(module, name="weight", amount=self_attn_rate)
                prune.remove(module, 'weight')
                print(f"剪枝后model.layers{layer_num}.self_attn.{proj}权重中非零参数的数量: {torch.sum(module.weight != 0)}")
            
        
        mlp_flag = bool(prune_conf['unstructure_local_prune_mlp_flag'])
        if mlp_flag:
            mlp_module_module_str = prune_conf['unstructure_local_prune_mlp_modules']
            mlp_list = comma_string_to_list(mlp_module_module_str)
            mlp_rate = float(prune_conf['unstructure_local_prune_mlp_rate'])
            for proj in mlp_list:
                module = getattr(model.layers[layer_num].mlp, proj)
                print(f"剪枝前model.layers{layer_num}.mlp.{proj}权重中非零参数的数量: {torch.sum(module.weight != 0)}")
                if prune_method == "L1Unstructured":
                    prune.l1_unstructured(module, name="weight", amount=mlp_rate)
                elif prune_method == "RandomStructured":
                    prune.random_unstructured(module, name="weight", amount=mlp_rate)
                prune.remove(module, 'weight')
                print(f"剪枝后model.layers{layer_num}.mlp.{proj}权重中非零参数的数量: {torch.sum(module.weight != 0)}")
        

    output_dir = prune_conf['unstructure_local_output_dir']
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("剪枝完成！")

















