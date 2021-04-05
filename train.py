"""Script used to train agents."""

import argparse
from pathlib import Path
import os
import tonic
from tonic.torch.agents import PPO
import matplotlib.pyplot as plt

cur_path = Path(__file__).absolute()
snapshots_path = Path('./results')
snapshots_path.mkdir(exist_ok=True)

# is_remote = not Path("/Users/samlerman").exists()
is_remote = False
debugged_logger = True
if is_remote:
    from clearml import Task
    if debugged_logger:
        from utils import logger

    task = Task.init(project_name="BIN", task_name="run", output_uri=str(snapshots_path))


def train(
        header, agent, environment, trainer, before_training, after_training,
        parallel, sequential, seed, name
):
    '''Trains an agent on an environment.'''

    # Capture the arguments to save them, e.g. to play with the trained agent.
    args = dict(locals())

    # Run the header first, e.g. to load an ML framework.
    if header:
        exec(header)

    # Build the agent.
    if isinstance(agent, str):
        agent = eval(agent)

    # Build the train and test environments.
    _environment = environment
    environment = tonic.environments.distribute(
        lambda: eval(_environment), parallel, sequential)
    test_environment = tonic.environments.distribute(
        lambda: eval(_environment))

    # Choose a name for the experiment.
    if hasattr(test_environment, 'name'):
        environment_name = test_environment.name
    else:
        environment_name = test_environment.__class__.__name__
    if not name:
        if hasattr(agent, 'name'):
            name = agent.name
        else:
            name = agent.__class__.__name__
        if parallel != 1 or sequential != 1:
            name += f'-{parallel}x{sequential}'

    # Initialize the logger to save data to the path environment/name/seed.
    path = os.path.join(environment_name, name, str(seed))
    tonic.logger.initialize(path, script_path=cur_path.absolute(), config=args)

    # Build the trainer.
    trainer = eval(trainer)
    trainer.initialize(
        agent=agent, environment=environment,
        test_environment=test_environment, seed=seed)

    # Run some code before training.
    if before_training:
        exec(before_training)

    # Train.
    trainer.run()

    # Run some code after training.
    if after_training:
        exec(after_training)


if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            return v

    def none_or_float(value):
        if value == 'None' or value is None:
            return None
        return float(value)

    def none_or_str(value):
        if value == 'None':
            return None
        return value

    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=none_or_str,
                        # default="AntBulletEnv-v0")
                        default="Walker2DBulletEnv-v0")
                        # default="HalfCheetahBulletEnv-v0")
    parser.add_argument("--model", type=none_or_str,
                        default="model")
    # default="HalfCheetahBulletEnv-v0")
    parser.add_argument('--seed', type=int,
                        default=0)
                        # default=2)
    args = parser.parse_args()
    if is_remote and debugged_logger:
        logger.initialize(task.get_logger())

    header = 'import tonic.torch; import sys; sys.path.append("{}")'.format(Path(__file__).parent.absolute())
    if args.model == "model" or args.model == "sigmoid_params_no_threshold":
        from Variants.model import actor_critic_bin
    elif args.model == "sigmoid_params_learned_threshold":
        from Variants.sigmoid_params_learned_threshold import actor_critic_bin
    elif args.model == "sigmoid_params_learned_threshold_bigger":
        from Variants.sigmoid_params_learned_threshold_bigger import actor_critic_bin
    elif args.model == "AE_Act_Adv":
        from Variants.AE_Act_Adv import actor_critic_bin
    elif args.model == "relu_strengths":
        from Variants.relu_strengths import actor_critic_bin
    agent = PPO(model=actor_critic_bin())
    # agent = PPO()
    environment = 'tonic.environments.Bullet("{}")'.format(args.env)
    trainer = 'tonic.Trainer(epoch_steps=50000, steps=int(1e5))'
    before_training = False
    after_training = False
    parallel = 1
    sequential = 1
    seed = args.seed
    name = "PPO_BIN_{}".format(args.model)
    # name = "PPO"
    print(name)

    os.chdir('./results')
    train(header, agent, environment, trainer, before_training, after_training, parallel, sequential, seed, name)
    for module in agent.model.actor.torso.modules():
        logger = module.compile_logs()
        x = [
            "inputs",
            "grads",
            "synapses"
        ]

        inputs = logger["inputs"],
        grads = logger["grads"],
        synapses = [logger["synapses"] * len(logger["grads"])]

        # todo figure out plotting
        ax = plt.subplot(111)
        ax.bar(x, inputs, width=0.2, color='b', align='center')
        ax.bar(x, grads, width=0.2, color='g', align='center')
        ax.bar(x, synapses, width=0.2, color='r', align='center')

        plt.savefig("plot.png")


