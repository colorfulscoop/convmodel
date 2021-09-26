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
            "streamlit>=0.88",
        ],
        "test": ["pytest>=5.0,<60", "black>=20.8b1,<21"],
    },
    version="0.2.0",
    author="Colorful Scoop",

    # Description info
    url="https://github.com/colorfulscoop/convmodel",
    description=(
        "convmodel provides a conversation model"
        " based on GPT-2."
    ),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",

    # Additional metadata
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
