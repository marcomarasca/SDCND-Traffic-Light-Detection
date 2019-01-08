import tensorflow as tf
import os
import io
import time
import glob

import numpy as np

from lxml import etree
from tqdm import tqdm
from PIL import Image

flags = tf.app.flags

flags.DEFINE_string('data_dir', None, 'Path to the folder where the images are stored')
flags.DEFINE_string('labels_dir', None, 'Path to the folder labels annotation are stored')
flags.DEFINE_string('model_path', None, 'Path to the frozen graph used for traffic light detection')
flags.DEFINE_string('label', None, 'The name of the corresponding label')
flags.DEFINE_integer('category_index', 10, 'The id of the traffic light category as detected by the model')

tf.app.flags.mark_flag_as_required('data_dir')
tf.app.flags.mark_flag_as_required('model_path')
tf.app.flags.mark_flag_as_required('label')

FLAGS = flags.FLAGS

def load_model(file_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(file_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph
            
def load_image(image_path):
    image = Image.open(image_path)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8), im_width, im_height

def run_inference(sess, ops, image_tensor, image):
    output_dict = {}
    
    time_s = time.time()
    num_detections, boxes, scores, classes = sess.run(ops, feed_dict={image_tensor: image})
    time_t = time.time() - time_s
    
    output_dict['num_detections'] = int(num_detections[0])
    output_dict['detection_classes'] = classes[0].astype(np.uint8)
    output_dict['detection_boxes'] = boxes[0]
    output_dict['detection_scores'] = scores[0]
    output_dict['detection_time'] = time_t
    
    return output_dict

def create_xml_annotation(detection_dict, label_path, label, category_index, threshold=0.5):
    root = etree.Element("annotation")
    
    etree.SubElement(root, "filename").text = os.path.basename(detection_dict['filename'])

    source = etree.SubElement(root, 'source')
    etree.SubElement(source, 'database').text = 'Unknown'
    
    size = etree.SubElement(root, 'size')
    width = detection_dict['width']
    height = detection_dict['height']

    etree.SubElement(size, 'width').text = str(width)
    etree.SubElement(size, 'height').text = str(height)
    etree.SubElement(size, 'depth').text = str(detection_dict['depth'])
    etree.SubElement(root, 'segmented').text = '0'

    num_detections = detection_dict['num_detections']
    detection_classes = detection_dict['detection_classes']
    detection_boxes = detection_dict['detection_boxes']
    detection_scores = detection_dict['detection_scores']

    # Selects the indexes that correspond to the correct category and that passes the detection score threshold
    traffic_lights_idx = np.where((detection_classes == category_index) & (detection_scores >= threshold))
    detection_boxes = detection_boxes[traffic_lights_idx]
    for box in detection_boxes:
        detection = etree.Element('object')
        etree.SubElement(detection, 'name').text = label
        etree.SubElement(detection, 'pose').text = 'Unspecified'
        etree.SubElement(detection, 'truncated').text = '0'
        etree.SubElement(detection, 'difficult').text = '0'
        bound_box = etree.SubElement(detection, 'bndbox')
        # Convert from normalized boxes
        etree.SubElement(bound_box, 'xmin').text = str(int(box[1] * width))
        etree.SubElement(bound_box, 'ymin').text = str(int(box[0] * height))
        etree.SubElement(bound_box, 'xmax').text = str(int(box[3] * width))
        etree.SubElement(bound_box, 'ymax').text = str(int(box[2] * height))
        root.append(detection)

    with open(label_path, 'wb') as f:
        f.write(etree.tostring(root, pretty_print=True))

def main(unused_argv):

    if FLAGS.labels_dir is None:
        FLAGS.labels_dir = os.path.join(FLAGS.data_dir, 'labels')

    if not os.path.isdir(FLAGS.labels_dir):
        os.makedirs(FLAGS.labels_dir)

    image_paths = glob.glob(os.path.join(FLAGS.data_dir, '*.jpg'))
    graph = load_model(FLAGS.model_path)

    with graph.as_default():
    
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        boxes_tensor = graph.get_tensor_by_name('detection_boxes:0')
        scores_tensor = graph.get_tensor_by_name('detection_scores:0')
        classes_tensor = graph.get_tensor_by_name('detection_classes:0')
        detections_tensor = graph.get_tensor_by_name('num_detections:0')

        ops = [detections_tensor, boxes_tensor, scores_tensor, classes_tensor]

        with tf.Session() as sess:
            for image_path in tqdm(image_paths, desc='Processing', unit=' images'):
                image, width, height = load_image(image_path)
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                # Actual detection.
                output_dict = run_inference(sess, ops, image_tensor, image_np_expanded)
                file_name = os.path.basename(image_path)
                # Adds some metadata
                output_dict['filename'] = file_name
                output_dict['width'] = width
                output_dict['height'] = height
                output_dict['depth'] = 3
                label_path = os.path.join(FLAGS.labels_dir, '{}.xml'.format(os.path.splitext(file_name)[0]))
                create_xml_annotation(output_dict, label_path, FLAGS.label, FLAGS.category_index)

if __name__ == '__main__':
  tf.app.run()
