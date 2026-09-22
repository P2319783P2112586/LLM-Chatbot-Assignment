import torch
import transformers

print("PyTorch version:", torch.__version__)
print("Transformers version:", transformers.__version__)
print("MPS available:", torch.backends.mps.is_available())
