'''Script used to plot results.'''

import argparse
import os
from pathlib import Path
is_remote = not Path("/Users/samlerman").exists()
if is_remote:
    from clearml import Task
import time
import matplotlib as mpl
import matplotlib.pyplot as plt
from tonic.plot import plot


snapshots_path = Path('./results')
snapshots_path.mkdir(exist_ok=True)
if is_remote:
    task = Task.init(project_name="SurF'N", task_name="plot", output_uri=str(snapshots_path))
os.chdir('./results')


if __name__ == '__main__':
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', nargs='+',
                        # default=["/Users/samlerman/Code/SurF'N/Original/results/Present/AntBulletEnv-v0"])
                        default=["/Users/samlerman/Code/BINs/results/Walker2DBulletEnv-v0"])
                        # default=["/Users/samlerman/Code/SurF'N/Original/results/Present/HalfCheetahBulletEnv-v0"])
    parser.add_argument('--x_axis', default='train/steps')
    parser.add_argument('--y_axis', default='test/episode_score')
    parser.add_argument('--x_label')
    parser.add_argument('--y_label')
    parser.add_argument('--interval', default='bounds')
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--show_seeds', type=bool, default=False)
    parser.add_argument('--columns', type=int)
    parser.add_argument('--x_min', type=int)
    parser.add_argument('--x_max', type=int)
    parser.add_argument('--baselines', nargs='+',
                        default="PPO")
                        # default="")
    parser.add_argument('--baselines_source', default='tensorflow')
    parser.add_argument('--name')
    parser.add_argument('--save_formats', nargs='*', default=['pdf', 'png'])
    parser.add_argument('--seconds', type=int, default=0)
    parser.add_argument('--cmap')
    parser.add_argument('--legend_columns', type=int)
    parser.add_argument('--font_size', type=int, default=12)
    parser.add_argument('--font_family', default='serif')
    parser.add_argument('--legend_font_size', type=int)
    parser.add_argument('--legend_marker_size', type=int, default=10)
    parser.add_argument('--backend', default=None)
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()

    # Backend selection, e.g. agg for non-GUI.
    has_gui = True
    if args.backend:
        mpl.use(args.backend)
        has_gui = args.backend.lower() != 'agg'
    del args.backend

    # Fonts.
    plt.rc('font', family=args.font_family, size=args.font_size)
    if args.legend_font_size:
        plt.rc('legend', fontsize=args.legend_font_size)
    del args.font_family, args.font_size, args.legend_font_size

    seconds = args.seconds
    del args.seconds

    # Plot.
    start_time = time.time()
    fig = plot(**vars(args), fig=None)

    try:
        # Wait until the Window is closed if GUI.
        if seconds == 0:
            if has_gui:
                while plt.get_fignums() != []:
                    plt.pause(0.1)

        # Repeatedly plot, waiting a few seconds until interruption.
        else:
            while True:
                if has_gui:
                    while time.time() - start_time < seconds:
                        plt.pause(0.1)
                        assert plt.get_fignums() != []
                else:
                    time.sleep(seconds)
                start_time = time.time()
                plot(**vars(args), fig=fig)
    except:
        pass
