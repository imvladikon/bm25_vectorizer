from setuptools import setup
import io
import os

__project_dir__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__reqs_path__ = os.path.join(__project_dir__, 'requirements.txt')
__readme_path__ = os.path.join(__project_dir__, 'README.md')

short_description = 'sklearn compatible BM25 vectorizer'

try:
    with io.open(__readme_path__, encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = short_description

try:
    with io.open(__reqs_path__, encoding='utf-8') as f:
        reqs = [line.strip() for line in f if not line.strip().startswith('#')]
except FileNotFoundError:
    reqs = []

setup(
    name='bm25_vectorizer',
    version='0.0.1',
    description=short_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Vladimir Gurevich',
    author_email='imvladikon@gmail.com',
    url="https://github.com/imvladikon/bm25_vectorizer",
    license='Apache2.0',
    py_modules=['bm25_vectorizer'],
    install_requires=reqs,
    extras_require={'dev': ['hypothesis']},
)
