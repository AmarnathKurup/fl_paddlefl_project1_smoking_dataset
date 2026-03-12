import paddle.fluid as fluid
from paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory

inputs = fluid.layers.data(name='x', shape=[25], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

fc = fluid.layers.fc(input=inputs, size=2, act='softmax')

cost = fluid.layers.cross_entropy(input=fc, label=label)
loss = fluid.layers.reduce_mean(cost)

startup_program = fluid.default_startup_program()

job_generator = JobGenerator()

optimizer = fluid.optimizer.SGD(learning_rate=0.01)

job_generator.set_optimizer(optimizer)
job_generator.set_losses([loss])
job_generator.set_startup_program(startup_program)

job_generator.set_infer_feed_and_target_names(['x'], [fc.name])

build_strategy = FLStrategyFactory()
build_strategy.fed_avg = True
build_strategy.inner_step = 1

strategy = build_strategy.create_fl_strategy()

job_generator.generate_fl_job(
    strategy,
    server_endpoints=["127.0.0.1:8181"],
    worker_num=2,
    output="fl_job_config"
)

print("FL job generated")