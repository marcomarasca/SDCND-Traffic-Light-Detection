import tensorflow as tf
from tensorflow.tools.graph_transforms import TransformGraph
import os
from distutils.version import LooseVersion

flags = tf.app.flags

flags.DEFINE_string('model_path', None, 'Path to the frozen model file')
flags.DEFINE_string('output_dir', None, 'Path to the folder where to save the optimized graph')

tf.app.flags.mark_flag_as_required('model_path')

FLAGS = flags.FLAGS

def load_graph(file_path):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(file_path, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name='')
    return graph_def

def graph_stats(graph_def):
    print('\nInput Feature Nodes: {}'.format([node.name for node in graph_def.node if node.op=='Placeholder']))
    print('Output Nodes: {}'.format([node.name for node in graph_def.node if ('detection' in node.name)]))
    print('Constant Count: {}'.format(len([node for node in graph_def.node if node.op=='Const'])))
    print('Identity Count: {}'.format(len([node for node in graph_def.node if node.op=='Identity'])))
    print('Total nodes: {}'.format(len(graph_def.node)))
    print('---------------------------')

def optimize_graph(model_file, output_dir, input_names = ['image_tensor'], output_names = ['num_detections', 'detection_classes', 'detection_scores', 'detection_boxes']):
    
    print('Optimizing model {}...'.format(model_file))
    
    graph_def = load_graph(model_file)
    
    graph_stats(graph_def)

    # TODO didn't test much of this operations
    if LooseVersion(tf.__version__) >= LooseVersion('1.12.0'):
        transforms = [
            'strip_unused_nodes(type=float, shape="1,299,299,3")',
            'remove_nodes(op=Identity, op=CheckNumerics)'
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
            'fold_old_batch_norms'
        ]
    else:
        print('[WARNING] Tensorflow version {} (< 1.12.0), some optimization disabled.'.format(tf.__version__))
        transforms = [
            'strip_unused_nodes(type=float, shape="1,299,299,3")',
            'remove_nodes(op=CheckNumerics)'
            'fold_constants(ignore_errors=true)',
            'fold_batch_norms',
            'fold_old_batch_norms'
        ]
    
    graph_def_optimized = TransformGraph(graph_def, input_names, output_names, transforms)
    
    print('\nAfter Optimization: ')
    graph_stats(graph_def_optimized)
    
    tf.train.write_graph(graph_def_optimized, logdir=output_dir, as_text=False, name='graph_optimized.pb')


def main(unused_argv):
    if FLAGS.output_dir is None:
        FLAGS.output_dir = os.path.join(os.path.dirname(FLAGS.model_path), 'optimized')
    
    optimize_graph(FLAGS.model_path, FLAGS.output_dir)

if __name__ == '__main__':
  tf.app.run()