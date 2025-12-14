import torch

print('CUDA可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print('GPU名称:', torch.cuda.get_device_name(device))
    print('GPU显存:', round(torch.cuda.get_device_properties(device).total_memory / (1024**3), 2), 'GB')
    print('sm版本:', torch.cuda.get_device_properties(device).major, '.', torch.cuda.get_device_properties(device).minor)
    print('bfloat16支持:', torch.cuda.is_bf16_supported())