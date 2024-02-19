from setuptools import setup

setup(name='flat-world',
      version='0.9.1',
      packages=['flat_world', 'flat_world.helpers', 'flat_world.ref_agents', 'flat_world.tasks',
                'flat_world.tasks.multi_agent'],
      package_data={'flat_world': ['artifacts/*.ttf']},
      install_requires=['torch', 'pytest', 'gym', 'numpy', 'tk', 'Pillow']
      )
