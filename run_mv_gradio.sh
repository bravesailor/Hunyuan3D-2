model_path='/data/models/Hunyuan3D-2mv'
sub_folder='hunyuan3d-dit-v2-mv'
texgen_model_path='/data/models/Hunyuan3D-2'
port=8081
device='cuda:1'
host='0.0.0.0'
queue_size=10
concurrency=1
# python gradio_app_mv_ff.py --model_path '/data/models/Hunyuan3D-2' --subfolder "hunyuan3d-dit-v2-0"  --port 8080 --device 'cuda:0'
python gradio_app_feature.py --queue_size ${queue_size} --concurrency ${concurrency} --texgen_model_path ${texgen_model_path} --model_path ${model_path} --subfolder ${sub_folder}  --port ${port} --host ${host} --device ${device} 