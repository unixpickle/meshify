import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# build custom rasterizer
# build with `python setup.py install`

sources = [
    'lib/custom_rasterizer_kernel/rasterizer.cpp',
]
extension_kwargs = {
    "extra_link_args": [f"-Wl,-rpath,{torch.utils.cpp_extension.TORCH_LIB_PATH}"],
}

if torch.cuda.is_available():
    extension_cls = CUDAExtension
    sources.append('lib/custom_rasterizer_kernel/rasterizer_gpu.cu')
    extension_kwargs["define_macros"] = [("WITH_CUDA", None)]
else:
    extension_cls = CppExtension

custom_rasterizer_module = extension_cls('custom_rasterizer_kernel', sources, **extension_kwargs)

setup(
    packages=find_packages(),
    version='0.1',
    name='custom_rasterizer',
    include_package_data=True,
    package_dir={'': '.'},
    ext_modules=[
        custom_rasterizer_module,
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
