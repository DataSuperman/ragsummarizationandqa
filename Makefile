help:
	@echo "make run - run the application"
	@echo "make venv_nuke - remove the virtual environment"
	@echo "make docker_notebook - run jupyter notebook in a docker container"
	@echo "make docker_run - build and run the docker container"

run: .venv
	export PYTHONPATH="$(PWD)"; \
	./.venv/bin/python3 modjo_task/run.py

venv_nuke:
	rm -rf .venv

.venv:
	which virtualenv || sudo apt-get install python3-virtualenv
	virtualenv -p python3 .venv
	./.venv/bin/pip install -r requirements.txt

install_llama:


docker_notebook:
	docker run -it --rm \
		-p 8888:8888 \
		-v "$(PWD):/home/jovyan/work" \
		jupyter/scipy-notebook


docker_run:
	docker build -t modjo_task .
	docker run -it --rm modjo_task