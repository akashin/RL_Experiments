import tensorflow as tf
from tensorflow.core.lib.core import error_codes_pb2
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

def main():
  server1 = server_lib.Server.create_local_server()
  server2 = server_lib.Server.create_local_server()
  server3 = server_lib.Server.create_local_server()
  cluster_def1 = cluster_pb2.ClusterDef()
  job1 = cluster_def1.job.add()
  job1.name = 'worker'
  job1.tasks[0] = server1.target[len('grpc://'):]
  job1.tasks[1] = server2.target[len('grpc://'):]

  cluster_def2 = cluster_pb2.ClusterDef()
  job2 = cluster_def2.job.add()
  job2.name = 'worker'
  job2.tasks[0] = server1.target[len('grpc://'):]
  job2.tasks[1] = server3.target[len('grpc://'):]

  config1 = config_pb2.ConfigProto(cluster_def=cluster_def1)
  config2 = config_pb2.ConfigProto(cluster_def=cluster_def2)

  with ops.Graph().as_default() as g:
    with ops.device('/job:worker/task:0'):
      var = variables.Variable(0, name='var')
      update_op = state_ops.assign_add(var, 1, name='var_assign_add')
      init = variables.global_variables_initializer()


  # with ops.Graph().as_default() as g1:
    # with ops.device('/job:worker1/task:1'):
      # var1 = variables.Variable(array_ops.zeros([2]), name='var1')
      # update_op1 = state_ops.assign_add(
          # var1, array_ops.ones([2]), name='var1_assign_add')
      # init1 = variables.global_variables_initializer()

  # with ops.Graph().as_default() as g2:
    # with ops.device('/job:worker2/task:1'):
      # var2 = variables.Variable(array_ops.zeros([2]), name='var2')
      # update_op2 = state_ops.assign_add(
          # var2, array_ops.ones([2]), name='var2_assign_add')
      # init2 = variables.global_variables_initializer()

  sess2 = session.Session(server2.target, graph=g, config=config1)
  sess3 = session.Session(server3.target, graph=g, config=config2)

  init.run(session=sess2)
  init.run(session=sess3)

  print(sess2.run(update_op))
  print(sess3.run(update_op))

  print(sess2.run(update_op))
  print(sess3.run(update_op))

if __name__ == "__main__":
  main()
