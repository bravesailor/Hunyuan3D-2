from PIL import Image

from hy3dgen.rembg import BackgroundRemover
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline

import trimesh
# model_path = 'tencent/Hunyuan3D-2'
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 指定使用第一块 GPU
# pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model_path)
# pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(model_path)
# device = "cuda:3"
external_model_v2_path='/data/models/Hunyuan3D-2'
v2_subfoler = "hunyuan3d-paint-v2-0-turbo"

# v2_subfoler = "hunyuan3d-dit-v2-0-turbo"
# pipeline_shapegen = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(external_model_v2_path, device)
pipeline_texgen = Hunyuan3DPaintPipeline.from_pretrained(external_model_v2_path)

# image_path = 'assets/demo.png'
# image_path = 'data/chain.jpg'
image_path = 'data/face1.png'
image = Image.open(image_path).convert("RGBA")
if image.mode == 'RGB':
    rembg = BackgroundRemover()
    image = rembg(image)

# mesh = pipeline_shapegen(image=image)[0]
# mesh = trimesh.load("data/chain.glb")
mesh = trimesh.load("data/face.glb")
mesh = pipeline_texgen(mesh, image=image)
mesh.export('demo.glb')
