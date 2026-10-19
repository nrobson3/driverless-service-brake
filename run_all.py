"""Run the whole analysis chain in order; every result lands in out/."""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ['sizing', 'trade_study', 'mechanism', 'make_poses', 'fea_run', 'hand_calcs', 'control_sim', 'can_codec']

if __name__ == '__main__':
    sys.path.insert(0, HERE)
    os.chdir(HERE)
    for name in STEPS:
        print('==', name)
        runpy.run_path(os.path.join(HERE, name + '.py'), run_name='__main__')
