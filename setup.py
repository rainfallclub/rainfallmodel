import os
import re
from setuptools import find_packages, setup


def get_version() -> str:
    with open(os.path.join("src", "rainfallmodel", "common", "version.py"), encoding="utf-8") as f:
        file_content = f.read()
        pattern = r"{}\W*=\W*\"([^\"]+)\"".format("VERSION")
        (version,) = re.findall(pattern, file_content)
        return version


def get_requires() -> list[str]:
    with open("requirements.txt", encoding="utf-8") as f:
        file_content = f.read()
        lines = [line.strip() for line in file_content.strip().split("\n") if not line.startswith("#")]
        return lines


def get_console_scripts() -> list[str]:
    console_scripts = ["rainfallmodel = rainfallmodel.cli:main"]
    if os.getenv("ENABLE_SHORT_CONSOLE", "1").lower() in ["true", "y", "1"]:
        console_scripts.append("rfm = rainfallmodel.cli:main")

    return console_scripts


extra_require = {
    "torch": ["torch>=2.0.0", "torchvision>=0.15.0"],
}


def main():
    setup(
        name="rainfallmodel",
        version=get_version(),
        author="rainfall",
        author_email="rainfallclub@163.com",
        description="A simple and powerful tool for building Large Language Models from scratch",
        long_description=open("README.md", encoding="utf-8").read(),
        long_description_content_type="text/markdown",
        keywords=["AI", "LLM", "RainFall", "Transformer","Pytorch"],
        license="Apache 2.0 License",
        url="todo",
        package_dir={"": "src"},
        packages=find_packages("src"),
        python_requires=">=3.9.0",
        install_requires=get_requires(),
        extras_require=extra_require,
        entry_points={"console_scripts": get_console_scripts()},
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "License :: OSI Approved :: Apache Software License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
    )


if __name__ == "__main__":
    main()
