import setuptools


setuptools.setup(
    name="convmodel",
    packages=setuptools.find_packages(),
    install_requires=[
        "pydantic>=1,<2",
        "transformers>=4.8.2,<5",
        "sentencepiece>=0.1.95,<0.2",
    ],
    extras_require={
        "train": [
            "pytorch-lightning>=1.4.1,<2",
            # Install pytorch-lightning as well as jsonargparse for enabling LightningCLI
            "jsonargparse[signatures]>=3.17.0,<4",
            "fire>=0.4.0,<0.5",
        ],
        "cli": [
            "streamlit>=0.87,<0.88",
        ],
        "test": ["pytest>=5.0,<60", "black>=20.8b1,<21"],
    },
    version="0.1.1",
    author="Colorful Scoop",
)
