import codecs
import os

from setuptools import setup


def read_file(filename):
    with codecs.open(os.path.join(os.path.dirname(__file__), filename), 'r', 'utf-8') as file:
        return file.read()


setup(name='flat-world',
      version='0.9.3',
      description='A text memory meant to be used with conversational language models.',
      long_description=read_file('README.md'),
      long_description_content_type='text/markdown',
      url='https://github.com/goodai-jose-solorzano/FlatWorld',
      packages=['flat_world', 'flat_world.helpers', 'flat_world.ref_agents', 'flat_world.tasks',
                'flat_world.tasks.multi_agent'],
      package_data={'flat_world': ['artifacts/*.ttf']},
      install_requires=['torch', 'pytest', 'gym', 'numpy', 'tk', 'Pillow']
      )
