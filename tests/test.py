# The sole purpose of this file is to check the GPU compatibity with torch and torchvision versions
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("NO GPU FOUND — running on CPU")

