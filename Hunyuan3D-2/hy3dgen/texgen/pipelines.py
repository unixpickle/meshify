# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import gc
import logging
import numpy as np
import os
import torch
from PIL import Image
from typing import List, Union, Optional


from .differentiable_renderer.mesh_render import MeshRender
from .utils.dehighlight_utils import Light_Shadow_Remover
from .utils.multiview_utils import Multiview_Diffusion_Net
from .utils.imagesuper_utils import Image_Super_Net
from .utils.uv_warp_utils import mesh_uv_wrap

logger = logging.getLogger(__name__)


def _choose_texgen_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Hunyuan3DTexGenConfig:

    def __init__(self, light_remover_ckpt_path, multiview_ckpt_path, subfolder_name, device=None):
        self.device = device or _choose_texgen_device()
        self.light_remover_ckpt_path = light_remover_ckpt_path
        self.multiview_ckpt_path = multiview_ckpt_path

        self.candidate_camera_azims = [0, 90, 180, 270, 0, 180]
        self.candidate_camera_elevs = [0, 0, 0, 0, 90, -90]
        self.candidate_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]

        self.render_size = 2048
        self.texture_size = 2048
        self.bake_exp = 4
        self.merge_method = 'fast'
        self.delight_steps = 50
        self.multiview_steps = 30

        self.pipe_dict = {'hunyuan3d-paint-v2-0': 'hunyuanpaint', 'hunyuan3d-paint-v2-0-turbo': 'hunyuanpaint-turbo'}
        self.pipe_name = self.pipe_dict[subfolder_name]


class Hunyuan3DPaintPipeline:
    @classmethod
    def from_pretrained(cls, model_path, subfolder='hunyuan3d-paint-v2-0-turbo', device=None):
        original_model_path = model_path
        if not os.path.exists(model_path):
            # try local path
            base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
            model_path = os.path.expanduser(os.path.join(base_dir, model_path))

            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, subfolder)

            if not os.path.exists(delight_model_path) or not os.path.exists(multiview_model_path):
                try:
                    import huggingface_hub
                    # download from huggingface
                    model_path = huggingface_hub.snapshot_download(
                        repo_id=original_model_path, allow_patterns=["hunyuan3d-delight-v2-0/*"]
                    )
                    model_path = huggingface_hub.snapshot_download(
                        repo_id=original_model_path, allow_patterns=[f'{subfolder}/*']
                    )
                    delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
                    multiview_model_path = os.path.join(model_path, subfolder)
                    return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder, device=device))
                except Exception:
                    import traceback
                    traceback.print_exc()
                    raise RuntimeError(f"Something wrong while loading {model_path}")
            else:
                return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder, device=device))
        else:
            delight_model_path = os.path.join(model_path, 'hunyuan3d-delight-v2-0')
            multiview_model_path = os.path.join(model_path, subfolder)
            return cls(Hunyuan3DTexGenConfig(delight_model_path, multiview_model_path, subfolder, device=device))
            
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.render = MeshRender(
            default_resolution=self.config.render_size,
            texture_size=self.config.texture_size,
            device=self.config.device)

    def load_models(self):
        self.get_delight_model()
        self.get_multiview_model()
        # self.get_super_model()

    def _clear_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.config.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()

    def _get_or_load_model(self, model_name):
        if model_name not in self.models:
            self._clear_memory()
            if model_name == 'delight_model':
                self.models[model_name] = Light_Shadow_Remover(self.config)
            elif model_name == 'multiview_model':
                self.models[model_name] = Multiview_Diffusion_Net(self.config)
            elif model_name == 'super_model':
                self.models[model_name] = Image_Super_Net(self.config)
            else:
                raise KeyError(f"Unknown model {model_name}")
        return self.models[model_name]

    def get_delight_model(self):
        return self._get_or_load_model('delight_model')

    def get_multiview_model(self):
        return self._get_or_load_model('multiview_model')

    def unload_model(self, model_name):
        model = self.models.pop(model_name, None)
        if model is None:
            return
        del model
        self._clear_memory()

    def unload_all_models(self):
        for model_name in list(self.models.keys()):
            self.unload_model(model_name)

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        self.get_delight_model().pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)
        self.get_multiview_model().pipeline.enable_model_cpu_offload(gpu_id=gpu_id, device=device)

    def run_delight(self, image, progress_callback=None):
        return self.get_delight_model()(image, progress_callback=progress_callback)

    def run_multiview(self, input_images, control_images, camera_info):
        return self.get_multiview_model()(input_images, control_images, camera_info)

    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True):
        normal_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map = self.render.render_normal(
                elev, azim, use_abs_coor=use_abs_coor, return_type='pl')
            normal_maps.append(normal_map)

        return normal_maps

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, return_type='pl')
            position_maps.append(position_map)

        return position_maps

    def bake_from_multiview(self, views, camera_elevs,
                            camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** self.config.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8

    def texture_inpaint(self, texture, mask):

        texture_np = self.render.uv_inpaint(texture, mask)
        texture = torch.tensor(texture_np / 255).float().to(texture.device)

        return texture

    def recenter_image(self, image, border_ratio=0.2):
        if image.mode == 'RGB':
            return image
        elif image.mode == 'L':
            image = image.convert('RGB')
            return image

        alpha_channel = np.array(image)[:, :, 3]
        non_zero_indices = np.argwhere(alpha_channel > 0)
        if non_zero_indices.size == 0:
            raise ValueError("Image is fully transparent")

        min_row, min_col = non_zero_indices.min(axis=0)
        max_row, max_col = non_zero_indices.max(axis=0)

        cropped_image = image.crop((min_col, min_row, max_col + 1, max_row + 1))

        width, height = cropped_image.size
        border_width = int(width * border_ratio)
        border_height = int(height * border_ratio)

        new_width = width + 2 * border_width
        new_height = height + 2 * border_height

        square_size = max(new_width, new_height)

        new_image = Image.new('RGBA', (square_size, square_size), (255, 255, 255, 0))

        paste_x = (square_size - new_width) // 2 + border_width
        paste_y = (square_size - new_height) // 2 + border_height

        new_image.paste(cropped_image, (paste_x, paste_y))
        return new_image

    @torch.no_grad()
    def __call__(self, mesh, image):

        if not isinstance(image, List):
            image = [image]

        images_prompt = []
        for i in range(len(image)):
            if isinstance(image[i], str):
                image_prompt = Image.open(image[i])
            else:
                image_prompt = image[i]
            images_prompt.append(image_prompt)
            
        images_prompt = [self.recenter_image(image_prompt) for image_prompt in images_prompt]

        images_prompt = [self.run_delight(image_prompt) for image_prompt in images_prompt]
        self.unload_model('delight_model')

        mesh = mesh_uv_wrap(mesh)

        self.render.load_mesh(mesh)

        selected_camera_elevs, selected_camera_azims, selected_view_weights = \
            self.config.candidate_camera_elevs, self.config.candidate_camera_azims, self.config.candidate_view_weights

        normal_maps = self.render_normal_multiview(
            selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
        position_maps = self.render_position_multiview(
            selected_camera_elevs, selected_camera_azims)

        camera_info = [(((azim // 30) + 9) % 12) // {-20: 1, 0: 1, 20: 1, -90: 3, 90: 3}[
            elev] + {-20: 0, 0: 12, 20: 24, -90: 36, 90: 40}[elev] for azim, elev in
                       zip(selected_camera_azims, selected_camera_elevs)]
        multiviews = self.run_multiview(images_prompt, normal_maps + position_maps, camera_info)
        self.unload_model('multiview_model')

        for i in range(len(multiviews)):
            # multiviews[i] = self.models['super_model'](multiviews[i])
            multiviews[i] = multiviews[i].resize(
                (self.config.render_size, self.config.render_size))

        texture, mask = self.bake_from_multiview(multiviews,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=self.config.merge_method)

        mask_np = (mask.squeeze(-1).cpu().numpy() * 255).astype(np.uint8)

        texture = self.texture_inpaint(texture, mask_np)

        self.render.set_texture(texture)
        textured_mesh = self.render.save_mesh()

        return textured_mesh
