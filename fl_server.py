from paddle_fl.core.server.fl_server import FLServer
from paddle_fl.core.master.fl_job import FLRunTimeJob

# Load FL runtime job
job = FLRunTimeJob()
job.load_server_job("fl_job_config", 0)

# Create server
server = FLServer()
server.set_server_job(job)

print("FL Server starting...")

# Start server
server.start()