# KokiKit
A General NeRF and Diffusion Toolkit for [KatUI](https://github.com/KokeCacao/KatUI)

## Installation
```bash
# for production
pip install "kokikit[full] @ git+https://github.com/KokeCacao/kokikit.git"

# if you only want diffusion stuff
pip install "kokikit[diffusion] @ git+https://github.com/KokeCacao/kokikit.git"

# if you only want nerf stuff
pip install "kokikit[nerf] @ git+https://github.com/KokeCacao/kokikit.git"

# if you only want nerf and reconstruction stuff
pip install "kokikit[reconstruction] @ git+https://github.com/KokeCacao/kokikit.git"

# for developers who intend to edit the module (but pylance will lint incorrectly)
git clone https://github.com/KokeCacao/kokikit.git
pip install -e .

# for developers who want local install (but changes won't automatically reflected)
git clone https://github.com/KokeCacao/kokikit.git
pip install .
```

## Citation

See [CREADITS.md](CREDITS.md) for all credits.

```
@misc{kokikit,
  title={KokiKit: A General NeRF and Diffusion Toolkit},
  author={Koke_Cacao},
  year={2023},
  howpublished={\url{https://github.com/KokeCacao/kokikit}},
  note={Open-source software}
}
```
