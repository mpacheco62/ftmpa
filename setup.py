import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ftmpa",
    version="0.0.1",
    author="Matias Pacheco",
    author_email="Matias.Pacheco.A@gmail.com",
    description="Fitting program for solids mechanics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mpacheco62/ftmpa",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3',
    install_requires=["scipy", "numpy"],
)