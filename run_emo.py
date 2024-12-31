import subprocess
import os

def set_params():
    # set common params

    audio_name = 'temp'
    audio_path = './../DATA/input_audio/'
    lm468_bs_np_path = './../DATA/lm468_bs_temp/lm468_bs.npy'
    
    # set params for EmoTalk
    params_emotalk = {
        "wav_path": f'{audio_path}{audio_name}.wav',
        "bs_dim": 52,
        "feature_dim": 832, 
        "period": 30,
        "device": "cuda",
        "model_path": "./pretrain_model/EmoTalk.pth",
        # "result_path": "./result/",
        "result_path": "./../DATA/output_EmoTalk/",
        
        "max_seq_len": 5000,
        "num_workers": 0,
        "batch_size": 1,
        "post_processing": True,
        
        # "blender_path": "./blender_ver_3_6/blender",
        "blender_path": "./blender/blender",
        # "python_render_path": "./render_mm_01.py",
        "python_render_path": "./render_mm.py",
        "blend_path": "./render.blend",
        
        "level": 1,
        "person": 0,
        "output_video": True,
        "bs52_level": 2,
        "lm468_bs_np_path": lm468_bs_np_path
    }
    return params_emotalk

def run_EmoTalk(params_emotalk, env):
    # print(f'Running EmoTalk with params: {params_emotalk}')
    print('-'*100)
    print('Start running EmoTalk...\n')
    
    # run EmoTalk(create_bs_face.py) as subprocess using subprocess.Popen
    cmd = ['python', 'create_bs_face.py']
    for key, value in params_emotalk.items():
        cmd.append(f'--{key}')
        cmd.append(str(value))
    
    p = subprocess.Popen(
        cmd,
        cwd='./EmoTalk_release',
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env
    )
    while p.poll() is None:
        line = p.stdout.readline()
        line = line.strip()
        if line:
            print(f'[EmoTalk] {line}')
    if p.returncode == 0:
        print('\nSubprogram EmoTalk success')
    else:
        print('\nSubprogram EmoTalk failed')

    print('-'*100)
    
def set_env():
    current_env = os.environ.copy()
    current_env['CUDA_VISIBLE_DEVICES'] = '0'
    new_ld_library_path = "/usr/local/cuda-11.7/lib64/stubs/:/usr/local/cuda-11.7/lib64:/usr/local/cuda-11.7/cudnn/lib"
    current_env['LD_LIBRARY_PATH'] = new_ld_library_path + ":" + current_env.get('LD_LIBRARY_PATH', '')
    return current_env

def main():
    # set params
    params_emotalk = set_params()
    env = set_env()
    
    # run programs
    run_EmoTalk(params_emotalk, env)
    # run_GeneFace(params_geneface, env)
    
def infer_video():
    # set params
    params_emotalk = set_params()
    env = set_env()
    # run programs
    run_EmoTalk(params_emotalk, env)
    # run_GeneFace(params_geneface, env)

if __name__ == "__main__":
    # infer_video()
    main()