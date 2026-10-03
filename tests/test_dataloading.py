import os
import pathlib
script_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(script_dir)
from . import common

def test_dataloading():
    for task in common.TASKS.values():
        common.get_data(task)
