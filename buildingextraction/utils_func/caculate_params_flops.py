# model_analyzer.py
"""
PyTorch模型分析工具
包含参数量计算、FLOPs计算、推理时间测试等功能
"""

import sys
sys.path.append('/public/home/tanyh_25/project/Mamba/ChangeMamba')
import torch
import torch.nn as nn

import argparse
import time
import math
from typing import Tuple, Dict, Any, Optional, Union

from changedetection.configs.config import get_config
from changedetection.models.ChangeMambaBE_fusion_enhancefre import ChangeMambaBEFusionEnhanceFre


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    计算模型参数量
    
    Args:
        model: PyTorch模型
        
    Returns:
        tuple: (总参数量, 可训练参数量)
    """
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    
    return total_params, trainable_params


def _ensure_int_tuple(shape: Union[Tuple, list, torch.Tensor]) -> Tuple[int, ...]:
    """确保输入形状是整数元组"""
    if isinstance(shape, torch.Tensor):
        # 如果是Tensor，转换为列表然后再转换为元组
        return tuple(int(x) for x in shape.tolist())
    elif isinstance(shape, (list, tuple)):
        return tuple(int(x) if hasattr(x, 'item') else int(x) for x in shape)
    else:
        raise ValueError(f"Shape must be tuple, list, or Tensor, got {type(shape)}")


def calculate_flops(model: nn.Module, input_shape: Union[Tuple[int, ...], list], device: str = 'cpu') -> int:
    """
    计算模型的FLOPs (浮点运算数)
    
    Args:
        model: PyTorch模型
        input_shape: 输入张量形状，例如 (1, 3, 224, 224)
        device: 计算设备
        
    Returns:
        int: FLOPs数量
    """
    # 确保输入形状格式正确
    input_shape = _ensure_int_tuple(input_shape)
    
    model = model.to(device)
    model.eval()
    
    total_flops = 0
    
    def flop_hook(module, input, output):
        nonlocal total_flops
        
        if isinstance(module, nn.Conv2d):
            # 卷积层FLOPs计算
            batch_size = input[0].size(0)
            output_height, output_width = output.size(2), output.size(3)
            output_dims = output_height * output_width
            kernel_dims = module.kernel_size[0] * module.kernel_size[1]
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups
            
            # 计算卷积操作的FLOPs
            conv_flops = (kernel_dims * in_channels // groups) * output_dims * out_channels * batch_size
            
            # bias操作的FLOPs
            if module.bias is not None:
                conv_flops += out_channels * output_dims * batch_size
                
            total_flops += int(conv_flops)
            
        elif isinstance(module, nn.Conv1d):
            # 1D卷积FLOPs
            batch_size = input[0].size(0)
            output_length = output.size(2)
            kernel_size = module.kernel_size[0]
            in_channels = module.in_channels
            out_channels = module.out_channels
            groups = module.groups
            
            conv_flops = (kernel_size * in_channels // groups) * output_length * out_channels * batch_size
            if module.bias is not None:
                conv_flops += out_channels * output_length * batch_size
                
            total_flops += int(conv_flops)
            
        elif isinstance(module, nn.Linear):
            # 线性层FLOPs
            batch_size = input[0].size(0)
            in_features = module.in_features
            out_features = module.out_features
            
            linear_flops = in_features * out_features * batch_size
            if module.bias is not None:
                linear_flops += out_features * batch_size
                
            total_flops += int(linear_flops)
            
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            # BatchNorm FLOPs: 归一化 + 缩放和偏移
            total_flops += int(2 * input[0].numel())
            
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm FLOPs
            total_flops += int(5 * input[0].numel())  # 均值、方差、归一化、缩放、偏移
            
        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU, nn.ELU, nn.GELU, nn.Sigmoid, nn.Tanh)):
            # 激活函数FLOPs
            total_flops += int(input[0].numel())
            
        elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
            # 池化操作FLOPs
            total_flops += int(input[0].numel())
            
        elif isinstance(module, nn.Dropout):
            # Dropout在推理时不产生FLOPs
            pass
    
    # 注册hooks
    hooks = []
    for module in model.modules():
        if not isinstance(module, nn.ModuleList) and not isinstance(module, nn.Sequential):
            hook = module.register_forward_hook(flop_hook)
            hooks.append(hook)
    
    # 执行前向传播
    input_tensor = torch.randn(input_shape).to(device)
    with torch.no_grad():
        _ = model(input_tensor)
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    return total_flops


def measure_inference_time(model: nn.Module, input_shape: Union[Tuple[int, ...], list], 
                          device: str = 'cpu', num_runs: int = 100, 
                          warmup_runs: int = 10) -> float:
    """
    测量模型推理时间
    
    Args:
        model: PyTorch模型
        input_shape: 输入张量形状
        device: 计算设备
        num_runs: 测试运行次数
        warmup_runs: 预热运行次数
        
    Returns:
        float: 平均推理时间(秒)
    """
    # 确保输入形状格式正确
    input_shape = _ensure_int_tuple(input_shape)
    
    model = model.to(device)
    model.eval()
    
    input_tensor = torch.randn(input_shape).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # 同步GPU
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    # 测量时间
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    
    return (end_time - start_time) / num_runs


def format_number(num: int) -> str:
    """
    格式化数字显示
    
    Args:
        num: 要格式化的数字
        
    Returns:
        str: 格式化后的字符串
    """
    if num >= 1e12:
        return f"{num/1e12:.2f}T"
    elif num >= 1e9:
        return f"{num/1e9:.2f}G"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    else:
        return str(num)


def analyze_model(model: nn.Module, input_shape: Union[Tuple[int, ...], list], 
                  device: str = 'cpu', compare_model: Optional[nn.Module] = None,
                  model_name: str = "Model") -> Dict[str, Any]:
    """
    全面分析模型性能
    
    Args:
        model: 要分析的模型
        input_shape: 输入形状
        device: 计算设备
        compare_model: 用于对比的模型(可选)
        model_name: 模型名称
        
    Returns:
        dict: 包含分析结果的字典
    """
    print(f"=== {model_name} Analysis ===")
    
    # 参数量分析
    total_params, trainable_params = count_parameters(model)
    print(f"Parameters:")
    print(f"  Total: {format_number(total_params)} ({total_params:,})")
    print(f"  Trainable: {format_number(trainable_params)} ({trainable_params:,})")
    
    # FLOPs分析
    try:
        flops = calculate_flops(model, input_shape, device)
        print(f"FLOPs: {format_number(flops)} ({flops:,})")
    except Exception as e:
        print(f"FLOPs calculation failed: {e}")
        flops = None
    
    # 推理时间分析
    try:
        inference_time = measure_inference_time(model, input_shape, device)
        print(f"Inference time: {inference_time*1000:.3f} ms")
    except Exception as e:
        print(f"Inference time measurement failed: {e}")
        inference_time = None
    
    # 准备返回结果
    results = {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'flops': flops,
        'inference_time': inference_time
    }
    
    # 与对比模型比较
    if compare_model is not None:
        print(f"\n=== Comparison ===")
        
        comp_total_params, comp_trainable_params = count_parameters(compare_model)
        print(f"Parameter ratio ({model_name}/Baseline): {total_params/comp_total_params:.2f}x")
        
        if flops is not None:
            try:
                comp_flops = calculate_flops(compare_model, input_shape, device)
                print(f"FLOPs ratio ({model_name}/Baseline): {flops/comp_flops:.2f}x")
                results['comp_flops_ratio'] = flops/comp_flops
            except:
                pass
        
        try:
            comp_inference_time = measure_inference_time(compare_model, input_shape, device)
            print(f"Speed ratio ({model_name}/Baseline): {inference_time/comp_inference_time:.2f}x")
            results['comp_speed_ratio'] = inference_time/comp_inference_time
        except:
            pass
    
    print()
    return results


def get_model_summary(model: nn.Module) -> Dict[str, int]:
    """
    获取模型结构摘要
    
    Args:
        model: PyTorch模型
        
    Returns:
        dict: 各类层的统计信息
    """
    layer_count = {}
    
    for name, module in model.named_modules():
        module_type = type(module).__name__
        if module_type in layer_count:
            layer_count[module_type] += 1
        else:
            layer_count[module_type] = 1
    
    return layer_count


def memory_usage(model: nn.Module, input_shape: Union[Tuple[int, ...], list], device: str = 'cpu') -> Dict[str, float]:
    """
    计算模型内存使用量
    
    Args:
        model: PyTorch模型
        input_shape: 输入形状
        device: 计算设备
        
    Returns:
        dict: 内存使用统计(MB)
    """
    # 确保输入形状格式正确
    input_shape = _ensure_int_tuple(input_shape)
    print(f"Debug: input_shape after conversion: {input_shape}, type: {type(input_shape)}")
    
    model = model.to(device)
    
    # 参数内存
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    
    # 缓冲区内存  
    buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024
    
    # 输入内存
    try:
        input_tensor = torch.randn(input_shape).to(device)
        input_memory = input_tensor.numel() * input_tensor.element_size() / 1024 / 1024
    except Exception as e:
        print(f"Error creating input tensor with shape {input_shape}: {e}")
        input_memory = 0
    
    return {
        'parameters_mb': param_memory,
        'buffers_mb': buffer_memory,
        'input_mb': input_memory,
        'total_mb': param_memory + buffer_memory
    }



def main():
    parser = argparse.ArgumentParser(description="Training on Building Extraction dataset")
    parser.add_argument('--cfg', type=str, default='/public/home/tanyh_25/project/Mamba/ChangeMamba/changedetection/configs/vssm1/vssm_small_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str, default='/public/home/tanyh_25/project/Mamba/ChangeMamba/pretrained_weight/vssm_small_0229_ckpt_epoch_222.pth')

    parser.add_argument('--dataset', type=str, default='ISBD')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--train_dataset_path', type=str, default='/public/home/cornhut/tyh/datasets/instance-segmentation-building-dataset-of-china/train')
    parser.add_argument('--train_data_list_path', type=str, default='/public/home/cornhut/tyh/datasets/instance-segmentation-building-dataset-of-china/train.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/public/home/cornhut/tyh/datasets/instance-segmentation-building-dataset-of-china/test')
    parser.add_argument('--test_data_list_path', type=str, default='/public/home/cornhut/tyh/datasets/instance-segmentation-building-dataset-of-china/test.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=500000)
    parser.add_argument('--model_type', type=str, default='ChangeMambaBE_Small')
    parser.add_argument('--model_param_path', type=str, default='/public/home/tanyh_25/project/Mamba/ChangeMamba/changedetection/saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)

    args = parser.parse_args()


def model_init(args):
    config = get_config(args)
    
    deep_model = ChangeMambaBEFusionEnhanceFre(
        output_building=2, 
        pretrained=args.pretrained_weight_path,
        patch_size=config.MODEL.VSSM.PATCH_SIZE, 
        in_chans=config.MODEL.VSSM.IN_CHANS, 
        num_classes=config.MODEL.NUM_CLASSES, 
        depths=config.MODEL.VSSM.DEPTHS, 
        dims=config.MODEL.VSSM.EMBED_DIM, 
        # ===================
        ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
        ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
        ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
        ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
        ssm_conv=config.MODEL.VSSM.SSM_CONV,
        ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
        ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
        ssm_init=config.MODEL.VSSM.SSM_INIT,
        forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
        # ===================
        mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
        mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
        mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
        # ===================
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        patch_norm=config.MODEL.VSSM.PATCH_NORM,
        norm_layer=config.MODEL.VSSM.NORM_LAYER,
        downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
        patchembed_version=config.MODEL.VSSM.PATCHEMBED,
        gmlp=config.MODEL.VSSM.GMLP,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    )

    return deep_model



if __name__ == "__main__":
    # 示例使用
    # 创建一个简单的测试模型
    parser = argparse.ArgumentParser(description="Training on Building Extraction dataset")
    parser.add_argument('--cfg', type=str, default='/public/home/tanyh_25/project/Mamba/ChangeMamba/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str, default='/public/home/tanyh_25/project/Mamba/ChangeMamba/pretrained_weight/vssm_tiny_0230_ckpt_epoch_262.pth')
    args = parser.parse_args()

    test_model = model_init(args)
    
    # 分析模型
    input_shape = torch.rand(1, 3, 256, 256)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    results = analyze_model(test_model, input_shape, device, model_name="TestModel")
    
    # 显示模型摘要
    print("Model Structure Summary:")
    summary = get_model_summary(test_model)
    for layer_type, count in summary.items():
        print(f"  {layer_type}: {count}")
    
    # 显示内存使用
    print("\nMemory Usage:")
    memory = memory_usage(test_model, input_shape, device)
    for key, value in memory.items():
        print(f"  {key}: {value:.2f} MB")
