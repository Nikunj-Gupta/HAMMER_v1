import os, glob


configs = "../../configs/random_seed_runs/"
files = glob.glob(configs+"*centralised*.yaml")

for file in files:
    filename = file.split("/")[-1].split(".")[0]
    if filename.split("-")[0] == "cn":
        code = "../../ppo_shared_network.py " if filename.split("-")[1] == "baseline" else "../../main_complete_state_and_prev_actions.py "
        code = "../../random_message.py " if filename.split("-")[1] == "random_message" else None
        code = "../../centralised_training.py " if filename.split('-')[1] == "centralised_training" else None
        with open(os.path.join(".", filename+".sh"), "w") as f:
            f.write("#!/bin/bash\n"
                    "#SBATCH --nodes=1\n"
                    "#SBATCH --gres=gpu:2\n"
                    "#SBATCH --ntasks-per-node=32\n"
                    "#SBATCH --mem=127000M\n"

                    "#SBATCH --time=2:00:00\n"
                    "#SBATCH --account=def-mtaylor3\n"
                    "tensorboard --logdir=logs/ --host 0.0.0.0 &\n")
            string = "python " + code + " " + "--config " + file

            f.write(string)
    else:
        code = "../../multiwalker.py "
        with open(os.path.join(".", filename + ".sh"), "w") as f:
            f.write("#!/bin/bash\n"
                    "#SBATCH --nodes=1\n"
                    "#SBATCH --gres=gpu:2\n"
                    "#SBATCH --ntasks-per-node=32\n"
                    "#SBATCH --mem=127000M\n"

                    "#SBATCH --time=24:00:00\n"
                    "#SBATCH --account=def-mtaylor3\n"
                    "tensorboard --logdir=logs/ --host 0.0.0.0 &\n")
            string = "python " + code + " " + "--config " + file

            f.write(string)








# dir = "."
# filename = "main-3agents-"
# exp = "../../main_complete_state_and_prev_actions.py"
# config_dir = "../../configs/random_seed_runs/main_3agents/"
# files = glob.glob(config_dir+"*.yaml")
# i=1
# check = list(range(21, 36))
# for file in files:
#     # if "more" not in str(file):
#     #     continue
#     with open(os.path.join(dir, filename+file.split("/")[-1].split(".")[0])+".sh", "w") as f:
#         f.write("#!/bin/bash\n"
#                 "#SBATCH --nodes=1\n"
#                 "#SBATCH --gres=gpu:2\n"
#                 "#SBATCH --ntasks-per-node=32\n"
#                 "#SBATCH --mem=127000M\n"
#                 "#SBATCH --time=10:00:00\n"
#                 "#SBATCH --account=def-mtaylor3\n"
#
#                 "tensorboard --logdir=logs/ --host 0.0.0.0 &\n")
#         string = "python " + exp + " " + "--config " + file
#         f.write(string)
#         i += 1
