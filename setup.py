from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        name="kokikit",
        version="0.0.1",
        description="A General NeRF and Diffusion Toolkit for KatachiUI",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/KokeCacao/kokikit",
        author="Koke_Cacao",
        author_email="i@kokecacao.me",
        packages=find_packages(),
        include_package_data=True,
        classifiers=[
            "Programming Language :: Python :: 3 ",
        ],
        keywords="utility",
        install_requires=[
            "torch",
            "diffusers",
            "einops",
            "numpy",
            "diffusers",
            # "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch",
            # "git+https://github.com/ashawkey/diff-gaussian-rasterization"
            # "git+https://gitlab.inria.fr/bkerbl/simple-knn"
        ],
        extras_require={
            "full": [
                "xformers",
                "tqdm",
                "rich",
            ],
        },
    )
