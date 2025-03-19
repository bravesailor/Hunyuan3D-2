#!/bin/bash
# python gradio_app_mv_ff.py --model_path '/data/models/Hunyuan3D-2' --subfolder "hunyuan3d-dit-v2-0"  --port 8080 --device 'cuda:0'

model_path='/data/models/Hunyuan3D-2'
sub_folder='hunyuan3d-dit-v2-0'
texgen_model_path='/data/models/Hunyuan3D-2'
port=8080
device='cuda:0'
host='0.0.0.0'
queue_size=5
concurrency=1
# python gradio_app_mv_ff.py --model_path '/data/models/Hunyuan3D-2' --subfolder "hunyuan3d-dit-v2-0"  --port 8080 --device 'cuda:0'
python gradio_app_mv_ff.py --queue_size ${queue_size} --concurrency ${concurrency} --model_path ${model_path} --subfolder ${sub_folder}  --port ${port} --host ${host} --device ${device} 