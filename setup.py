import setuptools


setuptools.setup(
    name="convmodel",
    packages=setuptools.find_packages(),
    install_requires=[
        "pydantic>=1,<2",
        "transformers>=4.8.2,<5",
        "sentencepiece>=0.1.95,<0.2",
        "fire>=0.4.0,<0.5",
    ],
    extras_require={
        "cli": [
            "streamlit>=0.87,<0.88",
        ],
        "test": ["pytest>=5.0,<60", "black>=20.8b1,<21"],
    },
    version="0.1.1",
    author="Colorful Scoop",
)
