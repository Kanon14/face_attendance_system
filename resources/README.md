# Solving issue with `dlib` Library

Some issue occurs during the pip installation regarding the `dlib` library

Solution: In the `venv` environment, cd to `resources`, manually install the Dlib compiled binary wheels based on Python version used.

```bash
.venv/Scripts/activate
cd ./resources/
uv pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
cd ..
```

Other Python compatibility: [Dlib Compiled Binary Wheels for Python on Windows & x64 CPUs](https://github.com/z-mahmud22/Dlib_Windows_Python3.x)