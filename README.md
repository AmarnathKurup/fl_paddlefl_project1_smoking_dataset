## Getting Started

Follow the steps below to run the PaddleFL federated learning project.

### 1. Clone the Repository


```bash
git clone https://github.com/AmarnathKurup/fl_paddlefl_project1_smoking_dataset.git
cd fl_paddlefl_project1_smoking_dataset

2. Start the docker 
Build the docker environment  : docker build -t paddlefl_env .

3. Run the docker container : docker run -it -v ${PWD}:/workspace paddlefl_env

4. Generate Federated Learning Job Configuration 
python fl_master.py this creates the fl_job_config folder

5. Start the server
python fl_server.py 

6. Start the trainer
python fl_trainer.py 0 and  python fl_trainer.py 1
