import sys
import paddle.fluid as fluid
from paddle_fl.core.trainer.fl_trainer import FLTrainer
from paddle_fl.core.master.fl_job import FLRunTimeJob
from reader import trainer_reader, test_reader

trainer_id = int(sys.argv[1])

job = FLRunTimeJob()
job.load_trainer_job("fl_job_config", trainer_id)

trainer = FLTrainer()
trainer.set_trainer_job(job)

trainer.start()

train_data = trainer_reader(trainer_id)

print("Trainer",trainer_id,"started")

for epoch in range(3):

    print("Trainer",trainer_id,"Epoch",epoch)

    for data in train_data:

        x,label = data

        feed={
            "x":x.reshape(1,25),
            "label":label.reshape(1,1)
        }

        trainer.run(feed=feed,fetch=[])

print("Training finished")

# evaluation
test_data = test_reader()

correct=0
total=0

for data in test_data:

    x,label=data

    result = trainer.run(
        feed={"x":x.reshape(1,25),"label":label.reshape(1,1)},
        fetch=[]
    )

    total+=1

print("Evaluation finished")