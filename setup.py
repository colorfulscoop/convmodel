import setuptools


setuptools.setup(
    name="torchlang",
    packages=setuptools.find_packages(),
    install_requires=[
    ],
    extras_require={
        "test": ["pytest>=5.0,<60", "black>=20.8b1,<21"],
    },
    version="0.0.0",
    author="Colorful Scoop",
)
