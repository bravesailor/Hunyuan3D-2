import torch
import torch.optim as optim
import numpy as np
from PIL import Image
from pygltflib import GLTF2
import io
from sklearn.cluster import KMeans

def color_quantize_deep(image_tensor, num_colors, num_steps=500, temperature=0.1, device='cuda'):
    """
    使用可微分深度学习方法进行颜色量化
    """
    # 使用k-means初始化调色板
    H, W = image_tensor.shape[2], image_tensor.shape[3]
    img_np = image_tensor.squeeze(0).permute(1,2,0).reshape(-1,3).cpu().numpy()
    kmeans = KMeans(n_clusters=num_colors, random_state=0).fit(img_np)
    palette_init = kmeans.cluster_centers_.astype(np.float32)
    
    # 初始化可训练参数
    palette = torch.tensor(palette_init, device=device).unsqueeze(0)
    palette.requires_grad_(True)
    
    optimizer = optim.Adam([palette], lr=0.005)
    
    # 优化循环
    for step in range(num_steps):
        current_temp = max(temperature * (0.97 ** step), 0.01)
        
        # 计算距离和权重
        pixels = image_tensor.permute(0,2,3,1).unsqueeze(3)  # (1, H, W, 1, 3)
        dists = torch.norm(pixels - palette.unsqueeze(1).unsqueeze(2), dim=4)
        weights = torch.softmax(-dists / current_temp, dim=-1)
        
        # 重建图像
        reconstructed = torch.sum(weights.unsqueeze(4) * palette.unsqueeze(1).unsqueeze(2), dim=3)
        reconstructed = reconstructed.permute(0,3,1,2)
        
        # 计算损失
        mse_loss = torch.mean((reconstructed - image_tensor)**2)
        tv_loss = torch.sum(torch.abs(reconstructed[:, :, :-1] - reconstructed[:, :, 1:])) + \
                 torch.sum(torch.abs(reconstructed[:, :, :, :-1] - reconstructed[:, :, :, 1:]))
        loss = mse_loss + 0.1 * tv_loss / (H*W*3)
        
        # 优化步骤
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f'Step {step}, Loss: {loss.item():.4f}, Temp: {current_temp:.3f}')

    # 硬分配
    with torch.no_grad():
        pixels = image_tensor.permute(0,2,3,1).unsqueeze(3)
        dists = torch.norm(pixels - palette.unsqueeze(1).unsqueeze(2), dim=4)
        indices = torch.argmin(dists, dim=3)
        quantized = palette[0, indices[0]]
    
    return quantized.cpu().numpy()

def process_glb(input_path, output_path, num_colors):
    """
    处理GLB文件的主函数
    """
    # 加载GLB文件
    glb = GLTF2.load(input_path)
    
    # 处理每个纹理图像
    for i, image in enumerate(glb.images):
        try:
            # 读取图像数据
            img_data = glb.get_image_data(image)
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            img_np = np.array(img) / 255.0
            
            # 转换为PyTorch张量
            img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2,0,1).unsqueeze(0).to('cuda')
            
            # 进行颜色量化
            quantized_np = color_quantize_deep(img_tensor, num_colors)
            quantized_np = (quantized_np * 255).astype(np.uint8)
            quantized_img = Image.fromarray(quantized_np)
            
            # 保存回GLB
            img_byte_arr = io.BytesIO()
            quantized_img.save(img_byte_arr, format='PNG')
            glb.images[i].uri = None
            glb.images[i].bufferView = glb.convert_image_to_buffer_view(img_byte_arr.getvalue(), 
                                                                        mime_type='image/png')
            print(f'Processed texture {i+1}/{len(glb.images)}')
        except Exception as e:
            print(f'Error processing texture {i}: {str(e)}')
    
    # 保存处理后的文件
    glb.save(output_path)
    print(f'Successfully saved processed model to {output_path}')

# 使用示例
if __name__ == "__main__":
    process_glb('input.glb', 'output.glb', num_colors=16)