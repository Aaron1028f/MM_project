import bpy
import os
import numpy as np
import sys

filename = str(sys.argv[-1])
root_dir = str(sys.argv[-2])

bs52_level = float(sys.argv[-3])
lm468_bs_np_path = str(sys.argv[-4])
output_video = bool(sys.argv[-5])

model_bsList = ["browDownLeft",
                "browDownRight",
                "browInnerUp",
                "browOuterUpLeft",
                "browOuterUpRight",
                "cheekPuff",
                "cheekSquintLeft",
                "cheekSquintRight",
                "eyeBlinkLeft",
                "eyeBlinkRight",
                "eyeLookDownLeft",
                "eyeLookDownRight",
                "eyeLookInLeft",
                "eyeLookInRight",
                "eyeLookOutLeft",
                "eyeLookOutRight",
                "eyeLookUpLeft",
                "eyeLookUpRight",
                "eyeSquintLeft",
                "eyeSquintRight",
                "eyeWideLeft",
                "eyeWideRight",
                "jawForward",
                "jawLeft",
                "jawOpen",
                "jawRight",
                "mouthClose",
                "mouthDimpleLeft",
                "mouthDimpleRight",
                "mouthFrownLeft",
                "mouthFrownRight",
                "mouthFunnel",
                "mouthLeft",
                "mouthLowerDownLeft",
                "mouthLowerDownRight",
                "mouthPressLeft",
                "mouthPressRight",
                "mouthPucker",
                "mouthRight",
                "mouthRollLower",
                "mouthRollUpper",
                "mouthShrugLower",
                "mouthShrugUpper",
                "mouthSmileLeft",
                "mouthSmileRight",
                "mouthStretchLeft",
                "mouthStretchRight",
                "mouthUpperUpLeft",
                "mouthUpperUpRight",
                "noseSneerLeft",
                "noseSneerRight",
                "tongueOut"]
# ================== Load the blend file ==================
obj = bpy.data.objects["face"]

# ================== Set up the scene ==================
bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.display.shading.light = 'MATCAP'
bpy.context.scene.display.render_aa = 'FXAA'
bpy.context.scene.render.resolution_x = int(512)
bpy.context.scene.render.resolution_y = int(768)
bpy.context.scene.render.fps = 30
bpy.context.scene.render.image_settings.file_format = 'PNG'

cam = bpy.data.objects['Camera']

cam.scale = [2, 2, 2]
bpy.context.scene.camera = cam

output_dir = root_dir + filename
blendshape_path = root_dir + filename + '.npy'

bs = np.load(blendshape_path) # bs has shape (num_frames, 52)

frame_num = bs.shape[0]

for i in range(frame_num):
    curr_bs = bs[i]
    for j in range(52):
        obj.data.shape_keys.key_blocks[model_bsList[j]].value = bs52_level * curr_bs[j]

    # render video if needed
    if output_video:
        bpy.context.scene.render.filepath = os.path.join(output_dir, '{}.png'.format(i))
        bpy.ops.render.render(write_still=True)