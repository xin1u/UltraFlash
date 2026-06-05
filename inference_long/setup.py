from setuptools import setup, find_packages

setup(
    name="ultraflash",
    version="1.0.0",
    description="Ultra Flash: Scaling Real-Time Streaming Video Generation to High Resolutions",
    author="Ultra Flash Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.4.0",
        "torchvision>=0.19.0",
        "einops",
        "omegaconf",
        "tqdm",
        "av>=13.1.0",
    ],
)
