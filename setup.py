from setuptools import setup, find_packages

setup(
    name="flow-grpo",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio",
        "setuptools==75.8.2",
        "transformers==4.57.3",
        "accelerate==1.4.0",
        "diffusers==0.39.0", 
        
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.15.2",
        "scikit-learn==1.6.1",
        "scikit-image==0.25.2",
        
        "albumentations==1.4.10",  
        "opencv-python==4.11.0.86",
        "pillow==10.4.0",
        
        "tqdm==4.67.1",
        "pydantic==2.10.6",  
        "requests==2.32.3",
        "matplotlib==3.10.0",
        
        # "flash-attn==2.7.4.post1",
        "deepspeed==0.16.4",  
        "peft==0.17.0",       
        "bitsandbytes==0.45.3",
        
        "aiohttp==3.11.13",
        "fastapi==0.115.11", 
        "uvicorn==0.34.0",
        
        "huggingface-hub==0.36.2",  
        "datasets==3.3.2",
        "tokenizers==0.22.2",
        
        "einops==0.8.1",
        "omegaconf==2.3.0",
        "qwen-vl-utils==0.0.11",
        "trl==0.8.6",
        "fire==0.7.0",
        "timm==1.0.15",
        "tensorboard==2.19.0",
        "rich==14.0.0",
        "nvidia-ml-py==12.570.86",
        "xformers",
        "absl-py",
        "ml_collections",
        "sentencepiece",
        "openai",
        "wandb",
        "pyyaml",
    ],
    extras_require={
        "dev": [
            "ipython==8.34.0",
            "black==24.2.0",
            "pytest==8.2.0"
        ]
    }
)
