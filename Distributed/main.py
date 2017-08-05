import argparse
import sys
import time

import tensorflow as tf

FLAGS = None

def main(_):
  print(FLAGS)

  ps_hosts = FLAGS.ps_hosts.split(",")

  # Create a cluster from the parameter server and worker hosts.
  if FLAGS.job_name == "ps":
    first_port = FLAGS.workers_start_port
    last_port = FLAGS.workers_start_port + FLAGS.n_workers
    worker_hosts = ["localhost:{}".format(port) for port in range(first_port, last_port)]
    cluster = tf.train.ClusterSpec({"ps": { FLAGS.task_index: ps_hosts[0]}, "worker": worker_hosts})
  elif FLAGS.job_name == "worker":
    port = FLAGS.workers_start_port + FLAGS.task_index
    cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": { FLAGS.task_index: "localhost:" + str(port) }})

  # Create and start a server for the local task.
  server = tf.train.Server(cluster,
      job_name=FLAGS.job_name,
      task_index=FLAGS.task_index)

  global_step = tf.contrib.framework.get_or_create_global_step()

  # Assigns ops to the local worker by default.
  with tf.device(tf.train.replica_device_setter(ps_tasks=1)):
    v = tf.Variable(42)

  if FLAGS.job_name == "ps":
    server.join()
  elif FLAGS.job_name == "worker":

    # Assigns ops to the local worker by default.
    with tf.device(tf.train.replica_device_setter(
      worker_device="/job:worker/task:%d" % FLAGS.task_index,
      cluster=cluster)):

      # Build model...
      # loss = tf.Variable(42)

      train_op = v.assign_add(1)
      # train_op = tf.train.AdagradOptimizer(0.01).minimize(
          # loss, global_step=global_step)

    # The StopAtStepHook handles stopping after running given steps.
    # hooks=[tf.train.StopAtStepHook(last_step=1000000)]
    hooks=[]

    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    is_chief = FLAGS.task_index == 0

    with tf.train.MonitoredTrainingSession(
        master=server.target,
        is_chief=is_chief,
        checkpoint_dir="/tmp/train_logs") as sess:
      while not sess.should_stop():
        # Run a training step asynchronously.
        # See `tf.train.SyncReplicasOptimizer` for additional details on how to
        # perform *synchronous* training.
        # sess.run handles AbortedError in case of preempted PS.
        if is_chief:
          print("I am chief and I do nothing")
        else:
          print("I am worker and I do work: {}".format(sess.run(train_op)))
        time.sleep(1)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  # Flags for defining the tf.train.ClusterSpec
  parser.add_argument(
      "--ps_hosts",
      type=str,
      default="",
      help="Comma-separated list of hostname:port pairs"
      )
  parser.add_argument(
      "--n_workers",
      type=int,
      default=100,
      help="Number of workers"
      )
  parser.add_argument(
      "--workers_start_port",
      type=int,
      default=6100,
      help="Start port for workers"
      )
  parser.add_argument(
      "--job_name",
      type=str,
      default="",
      help="One of 'ps', 'worker'"
      )
  # Flags for defining the tf.train.Server
  parser.add_argument(
      "--task_index",
      type=int,
      default=0,
      help="Index of task within the job"
      )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
