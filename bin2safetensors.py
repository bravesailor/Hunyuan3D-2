import torch
import safetensors.torch
# weight = torch.load("/data/models/Hunyuan3D-2/hunyuan3d-paint-v2-0-turbo/vae/diffusion_pytorch_model.safetensors")
weight = torch.load("/data/models/Hunyuan3D-2/hunyuan3d-paint-v2-0-turbo/vae/diffusion_pytorch_model.bin")

safetensors.torch.save_file(weight, "/data/models/Hunyuan3D-2/hunyuan3d-paint-v2-0-turbo/vae/diffusion_pytorch_model.safetensors")