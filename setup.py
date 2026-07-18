from setuptools import setup, find_packages

setup(
    name="advanced-small-object-detection",
    version="1.0.0",
    description="Advanced Small Object Detection with P2 Heads, High-Resolution Training, and SAHI Inference",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "ultralytics>=8.2.0",
        "sahi>=0.11.18",
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "pycocotools>=2.0.8",
        "opencv-python>=4.9.0",
        "Pillow>=10.2.0",
        "numpy>=1.26.0",
        "pyyaml>=6.0.1",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "plotly>=5.18.0",
        "tqdm>=4.66.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
