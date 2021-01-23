all:
		clear
# 		PYTHONPATH=$(PWD) python3 multiwalker.py --config configs/multiwalker/config.yaml
get_logs: 
		clear 
		scp -r nikunjg@graham.computecanada.ca:/home/nikunjg/projects/def-mtaylor3/nikunjg/HAMMER/runs/2021/newruns/logs /home/nikunj/work/HAMMER/runs/2021/newruns 
		scp -r nikunjg@graham.computecanada.ca:/home/nikunjg/projects/def-mtaylor3/nikunjg/HAMMER/runs/2021/runs-continuous-mes/logs /home/nikunj/work/HAMMER/runs/2021/runs-continuous-mes 


tensorboard: 
		ssh -N -f -L localhost:6006:gra102:6006 nikunjg@graham.computecanada.ca 

plot: 
	clear 
	python3 plots.py 
