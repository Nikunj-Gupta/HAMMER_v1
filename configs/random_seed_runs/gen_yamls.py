import yaml
import os

# domain = "cn"
# cn = 1
# exp = "baseline"
# main = 0
# n = 5
# rs = [10, 16, 24]
# config_script_path = "../ppo_shared_network/config_script.yaml"
# dump_dir_path = "."


# domain = "cn"
# cn = 1
# exp = "main"
# main = 1
# n = 5
# rs = [10, 16, 24]
# config_script_path = "../main_complete_state_and_previous_actions/config_script.yaml"
# dump_dir_path = "."
#
#
# domain = "mw"
# exp = "baseline"
# n = 5
# rs = [10, 11, 15]
# main = 0
# cn = 0
# config_script_path = "../multiwalker/config_script.yaml"
# dump_dir_path = "."

# domain = "mw"
# exp = "main"
# n = 5
# rs = [10, 11, 15]
# main = 1
# cn = 0
# config_script_path = "../multiwalker/config_script.yaml"
# dump_dir_path = "."

# domain = "cn"
# exp = "random_message"
# n = 3
# rs = [10, 16, 24]
# main = 0
# cn = 1
# config_script_path = "../random_message/config_script.yaml"
# dump_dir_path = "."


domain = "cn"
exp = "centralised_training"
n = 3
rs = [10, 16, 24]
main = 0
cn = 1
config_script_path = "../ppo_shared_network/config_script.yaml"
dump_dir_path = "."


with open(config_script_path) as f:
    data = yaml.load(f)

for i in rs:

    filename = '-'.join([domain, exp, "agents_"+str(n), "rs_"+str(i)])

    if cn:
        if main:
            data["main"]["exp_name"] = filename
            data["main"]["random_seed"] = i
        else:
            data["exp_name"] = filename
            data["random_seed"] = i
    else:
        data["main"]["exp_name"] = filename
        data["main"]["random_seed"] = i
        data["main"]["n_walkers"] = int(n)
        data["main"]["message_len"] = 15
        data["main"]["max_episodes"] = 10000
        data["main"]["main"] = main

    filename = filename + ".yaml"
    with open(os.path.join(dump_dir_path, filename), "w") as f:
        yaml.dump(data, f)








#
# dir = "multiwalker"
# filename = "config_baseline.yaml"
# with open(os.path.join(dir, filename)) as f:
#     data = yaml.load(f)
#
# for i in range(16):
#     data["main"]["exp_name"] = dir + "-baseline-nwalkers-5-random_seed-" + str(i)
#     temp = data["main"]["exp_name"]
#     data["main"]["random_seed"] = i
#     data["main"]["n_walkers"] = 5
#     data["main"]["message_len"] = 8
#     data["main"]["max_episodes"] = 10000
#
#     data["main"]["main"] = 0
#
#     filename = "config-baseline-nwalkers-5-rs-" + str(i)+".yaml"
#     # for j in meslen:
#     #     data["main"]["exp_name"] = temp + "-meslen" + str(j)
#     #     data["main"]["message_len"] = j
#     #     filename = "config-rs-"+str(i)+"-meslen"+str(j)+".yaml"
#     with open(os.path.join(dir, filename), "w") as f:
#         yaml.dump(data, f)
#
#

# for i in range(16):
#     data["main"]["random_seed"] = i
#     data["main"]["exp_name"] = temp + str(i)
#     data["main"]["max_episodes"] = 100000
#     filename = "config-rs-"+str(i)+".yaml"
#     with open(os.path.join(dir, filename), "w") as f:
#         yaml.dump(data["main"], f)