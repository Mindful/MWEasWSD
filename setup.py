import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="resolve",
    version="0.1",
    author="Joshua Tanner, Jacob Hoffman",
    author_email="mindful.jt@gmail.com",
    description="Look up words and multi-word expressions in context",
    keywords=['WSD', 'word sense disambiguation' 'MWE', 'dictionary'],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Mindful/MWEasWSD",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.9',
    install_requires=[
        "torch~=1.12",
        "transformers~=4.1",
        "pytorch-lightning~=1.5",
        "torchmetrics~=0.7",
        "tqdm",
        "nltk~=3.6",
        "gdown",
        "gitpython",
        "jsonlines",
        "matplotlib",
        "wandb~=0.12",
        "jupyter~=1.0",
        "lxml~=4.8",
        "scikit-learn~=1.1",
        "fugashi~=1.1",
        "ipadic==1.0.0",
        "unidic-lite~=1.0"
    ]
)
