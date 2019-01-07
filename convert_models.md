## Steps used to convert the latest Tensorflow models to be compatible with tensorflow 1.3.0

1. Download models from latest tensorflow model zoo and relative configurations:
    
    https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
    https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs

2. Create conda env for tensorflow 1.4: 
    
    conda create -n tf1.4 python=3.6
    conda activate tf1.4

3. Install tensorflow 1.4.0:

    pip install tensorflow==1.4.0

3. Install dependencies:

    pip install pillow lxml matplotlib

4. Clone tf models repo:

    git clone https://github.com/tensorflow/models.git temp

5. Checkout compatible version:

    cd temp
    git checkout d135ed9c04bc9c60ea58f493559e60bc7673beb7

6. Copy temp/research/object_detection and temp/research/slim to exporter

7. Download protoc and extract the protoc.exe into exporter:

    https://github.com/protocolbuffers/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip

8. Move to exporter folder:

    cd exporter

9.  Compile proto buffers:

    protoc.exe object_detection/protos/*.proto --python_out=.

10. Set PYTHONPATH:

    SET PYTHONPATH=%cd%;%cd%\slim

11. Run tests:

    python object_detection/builders/model_builder_test.py

12. Export the model(s):

    python object_detection\export_inference_graph.py --input_type image_tensor --pipeline_config_path ../config/ssd_inception_v2_coco.config --trained_checkpoint_prefix ../models/ssd_inception_v2_coco_2018_01_28/model.ckpt --output_directory models/ssd_inception_v2_coco

    python object_detection\export_inference_graph.py --input_type image_tensor --pipeline_config_path ../config/faster_rcnn_inception_v2_coco.config --trained_checkpoint_prefix ../models/faster_rcnn_inception_v2_coco_2018_01_28/model.ckpt --output_directory models/faster_rcnn_inception_v2_coco

