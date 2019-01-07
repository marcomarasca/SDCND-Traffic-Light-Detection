# Traffic Light Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The repository contains the instructions and datasets for the pipeline used in training an *object detector* for traffic lights to be integrated into the final [Capstone project](https://github.com/aviralksingh/CarND-SuperAI-Capstone) of the Udacity Self Driving Car Nanodegree.

The project requires to detect traffic lights from the image send by the onboard camera and classify the detections into the 3 available categories: green, yellow and red so that the car can decide on how to behave in proximity of traffic lights.

One of the possible approaches would be to first run an object detector and then run a classifier, while a good solution this would require to train 2 different models and run them both in sequence. This may add overhead as the performance penalty to run the separate classifier, even though small might affect the driving behaviour.

The approach we took is instead to train an end-to-end model on an object detection pipeline, treating each traffic light state as a separate class, this approach comes from the the work done by [Bosch]((https://github.com/bosch-ros-pkg/bstld)) on their Small Traffic Light Dataset.

We use the [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection) in order to fine-tune the available models already trained on the [COCO Dataset](http://cocodataset.org/).

Dataset
---

Various datasets for the tasks are available, including the [Bosch Small Traffic Light Dataset](https://hci.iwr.uni-heidelberg.de/node/6132). For this task we decided to perform transfer learning on some well known models on a small set of manually labelled images that were captured in different conditions (simulator and real world).

Some publicly available datasets from other students working on the same project can also be used for validation or training, including the dataset provided by [ColdKnight](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset).

The images were labelled with [LabelImg](https://github.com/tzutalin/labelImg) and can be found in this repository with their annotations.

The repository contains a small [utility](./create_tf_record.py) (loosely based on the [TensorFlow object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection) tool) that converts the annotated images into a [TensorFlow Record](https://www.tensorflow.org/tutorials/load_data/tf-records). For more details on TF Records see https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md.

Training
---

In order to train the model we use the [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection), in the following the steps that are needed to fine-tune an trained model:

1. Download the trained models from the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) with their associated [pipeline configuration](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).

2. Download the [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection) and perform the required [installation steps](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md):
   * ```git clone https://github.com/tensorflow/models.git```
   * Copy the ```research\object_detection``` and ```research\slim``` folders to the root folder
   * Get a copy of the [Protobuff Compiler](https://github.com/protocolbuffers/protobuf/) (e.g. For [Windows](https://github.com/protocolbuffers/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip))
   * Compile the protcol buffers: ```protoc.exe object_detection/protos/*.proto --python_out=.```
   * Set your PYTHONPATH: ```SET PYTHONPATH=%cd%;%cd%\slim```
   * Test that the API is installed: ```python object_detection/builders/model_builder_test.py```

3. [Configure the pipeline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md), copies of the configurations used can be found in the [config](./config) folder.

4. Install the [COCO Api](https://github.com/cocodataset/cocoapi), if you run into ["ImportError: No module named 'pycocotools'"](https://github.com/matterport/Mask_RCNN/issues/6):

    ```
    pip install Cython
    pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
    ```

    If you run into ["TypeError: can't pickle dict_values objects"](https://github.com/tensorflow/models/issues/4780) look into object_detection\model_lib.py for ```category_index.values()``` and replace with ```list(category_index.values())```


4. [Run](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md) the training session:

    ```
    python object_detection\model_main.py --pipeline_config_path=path/to/the/model/config --model_dir=path/to/the/output
    ```

5. Watch it happen with tensorboard:

        tensorboard --logdir=path/to/the/output


    And open the browser to `http://{machine_ip}:6006`

