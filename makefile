all:
		clear
# 		PYTHONPATH=$(PWD) python3 multiwalker.py --config configs/multiwalker/config.yaml
get_logs: 
		clear 
		# scp -r nikunjg@graham.computecanada.ca:/home/nikunjg/projects/def-mtaylor3/nikunjg/HAMMER/runs/2021/mw/logs /home/nikunj/work/HAMMER/runs/2021/mw 
		# scp -r nikunjg@graham.computecanada.ca:/home/nikunjg/projects/def-mtaylor3/nikunjg/HAMMER/runs/2021/test/logs /home/nikunj/work/HAMMER/runs/2021/test 
		scp -r nikunjg@graham.computecanada.ca:/home/nikunjg/projects/def-mtaylor3/nikunjg/HAMMER/runs/2021/meslen-runs/logs /home/nikunj/work/HAMMER/runs/2021/meslen-runs 

tensorboard: 
		ssh -N -f -L localhost:6006:gra902:6006 nikunjg@graham.computecanada.ca 

