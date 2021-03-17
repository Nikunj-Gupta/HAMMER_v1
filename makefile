all:
		clear
# 		PYTHONPATH=$(PWD) python3 multiwalker.py --config configs/multiwalker/config.yaml
get_logs: 
		clear 
		scp -r nikunjg@graham.computecanada.ca:/home/nikunjg/projects/def-mtaylor3/nikunjg/HAMMER/runs/2021/randommes/logs /home/nikunj/work/HAMMER/runs/2021/randommes 


tensorboard: 
		ssh -N -f -L localhost:6006:gra102:6006 nikunjg@graham.computecanada.ca 

plot: 
	clear 
	python3 plots.py 

gen_plots: 

	clear 
	python3 merge_randomseeds_and_plot.py 
	# python3 check.py 



ddpg: 
	clear 
	python3 hammer-cn-ddpg.py --agent_alg DDPG --meslen 1 --log_dir test/logs/hammer-1 

hammer: 
	clear 
	time python3 hammer-cn.py \
	--config configs/2021/cn/hyperparams.yaml \
	--maxepisodes 100000 \
	--expname hammer-test-run \
	--hammer 1 \
	--randomseed 10 \
	--discretemes 0 \
	--meslen 1 


analyze: 

	clear 
	python3 new_comm_analysis.py \
	--load runs/2021/message_gradients_rs/save-dir/500000_Hammerwithgradients--cn--partialobs0--meslen1--dru--rs10/ --partialobs 0 --dru_toggle 1 \
	--expname commanaly--CN--dru --maxepisodes 1 --maxcycles 50 

	python3 new_comm_analysis.py \
	--load runs/2021/message_gradients_rs/save-dir/500000_Hammerwithgradients--cn--partialobs0--meslen1--nodru--rs10/ --partialobs 0 --dru_toggle 0 \
	--expname commanaly--CN--nodru --maxepisodes 1 --maxcycles 50 


	python3 new_comm_analysis.py \
	--load runs/2021/message_gradients_rs/save-dir/500000_Hammerwithgradients--cn--partialobs1--meslen1--dru--rs10 --partialobs 1 --dru_toggle 1 \
	--expname commanaly--ModifiedCN--dru --maxepisodes 1 --maxcycles 50 

	python3 new_comm_analysis.py \
	--load runs/2021/message_gradients_rs/save-dir/500000_Hammerwithgradients--cn--partialobs1--meslen1--nodru--rs10 --partialobs 1 --dru_toggle 0 \
	--expname commanaly--ModifiedCN--nodru --maxepisodes 1 --maxcycles 50 

gen_runs: 
	clear 
	python3 gen_new_cn_runs.py 

render: 
	clear 
	python3 render.py \
	--load_hammer runs/2021/message_gradients_rs/save-dir/500000_Hammerwithgradients--cn--partialobs0--meslen1--dru--rs10/ \
	--load_il runs/2021/message_gradients_rs/save-dir/500000_ILsharedparams--cn--partialobs0--rs10/ \
	--partialobs 0 --dru_toggle 1 \
	--expname commanaly--CN--dru --maxepisodes 50 --maxcycles 50 

NUMBERS = 5 15 25 

agents = 2 
nodrumeslen = 2 3 4 5 
drumeslen = 1 2 3 4 
seeds = 999 

guesser: 
	clear 
	$(foreach var,$(nodrumeslen),python3 hammer-number-guessing.py --nagents 2 --dru_toggle 0 --meslen $(var) --randomseed 99;) 
	# $(foreach var,$(nodrumeslen),python3 hammer-number-guessing.py --nagents 2 --dru_toggle 0 --meslen $(var) --randomseed 10;) 

	# $(foreach var,$(drumeslen),python3 hammer-number-guessing.py --nagents 2 --dru_toggle 1 --meslen $(var) --randomseed 99;) 
	
sum-guesser-analyze: 
	clear 
	python3 sum_guessing_comm_analysis.py --dru_toggle 0 \
	--load sumguesser-save-dir-new/0_guesser--nagents2--dru0--meslen0--rs--99 --meslen 0 --nagents 2 
	# python3 sum_guessing_comm_analysis.py --dru_toggle 1 \
	# --load sumguesser-save-dir-new/50000_guesser--nagents2--dru1--meslen1--rs--99 --meslen 1 --nagents 2 

sumguesser-3agents: 
	clear 
	# python3 hammer-number-guessing.py --nagents 3 --dru_toggle 0 --meslen 1 --randomseed 99 
	# python3 hammer-number-guessing.py --nagents 3 --dru_toggle 0 --meslen 0 --randomseed 99 
	python3 hammer-number-guessing.py --nagents 3 --dru_toggle 1 --meslen 1 --randomseed 99 

sumguesser-analyze-3agents: 
	clear 
	python3 sum_guessing_comm_analysis.py --dru_toggle 0 \
	--load sumguesser-save-dir-3agents/50000_guesser--nagents3--dru0--meslen1--rs--99 --meslen 1 --nagents 3 
	# python3 sum_guessing_comm_analysis.py --dru_toggle 1 \
	# --load sumguesser-save-dir-new/50000_guesser--nagents2--dru1--meslen1--rs--99 --meslen 1 --nagents 2 

sumgame-hardcode: 
	python3 hammer-number-guessing.py --nagents 3 --dru_toggle 0 --meslen 1 --randomseed 99 --envname harcoded_messages_guesser 

multiplygame-2agents: 
	python3 hammer-number-guessing.py --nagents 2 --dru_toggle 0 --meslen 0 --randomseed 99 --envname multiply_guesser 
	python3 hammer-number-guessing.py --nagents 2 --dru_toggle 0 --meslen 1 --randomseed 99 --envname multiply_guesser 
	python3 hammer-number-guessing.py --nagents 2 --dru_toggle 1 --meslen 1 --randomseed 99 --envname multiply_guesser 

multiplygame-3agents: 
	python3 hammer-number-guessing.py --nagents 3 --dru_toggle 0 --meslen 0 --randomseed 99 --envname multiply_guesser 
	python3 hammer-number-guessing.py --nagents 3 --dru_toggle 0 --meslen 1 --randomseed 99 --envname multiply_guesser 
	python3 hammer-number-guessing.py --nagents 3 --dru_toggle 1 --meslen 1 --randomseed 99 --envname multiply_guesser 

multiplyguesser-analyze-2agents: 
	clear 
	python3 sum_guessing_comm_analysis.py --dru_toggle 0 \
	--load sumguesser-save-dir-3agents/50000_multiply_guesser--nagents2--dru0--meslen1--rs--99 --meslen 1 --nagents 2 
	# python3 sum_guessing_comm_analysis.py --dru_toggle 1 \
	# --load sumguesser-save-dir-new/50000_guesser--nagents2--dru1--meslen1--rs--99 --meslen 1 --nagents 2 
