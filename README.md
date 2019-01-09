# Traffic Light Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
The repository contains the instructions and links to the datasets for the pipeline used in training an *object detector* for traffic lights to be integrated into the final [Capstone project](https://github.com/aviralksingh/CarND-SuperAI-Capstone) of the Udacity Self Driving Car Nanodegree.

The project requires to detect traffic lights from the image send by the onboard camera and classify the detections into the 3 available categories: green, yellow and red so that the car can decide on how to behave in proximity of traffic lights.

One of the possible approaches would be to first run an object detector and then run a classifier, while a good solution this would require to train 2 different models and run them both in sequence. This may add overhead as the performance penalty to run the separate classifier, even though small might affect the driving behaviour.

The approach we took is instead to train an end-to-end model on an object detection pipeline, treating each traffic light state as a separate class, this approach comes from the the work done by [Bosch]((https://github.com/bosch-ros-pkg/bstld)) on their Small Traffic Light Dataset.

We use the [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection) in order to fine-tune the available models already trained on the [COCO Dataset](http://cocodataset.org/).

Dataset
---

TLDR; The datasets used for training can be downloaded from [here](https://drive.google.com/open?id=1NXqHTnjVC1tPjAB5DajGc30uWk5VPy7C).

Other datasets for the tasks are available, including the [Bosch Small Traffic Light Dataset](https://hci.iwr.uni-heidelberg.de/node/6132) and publicly available datasets from other students working on the same project can also be used for validation or training, including the dataset provided by [ColdKnight](https://github.com/coldKnight/TrafficLight_Detection-TensorFlowAPI#get-the-dataset).

For this task we decided to perform transfer learning on some [well known models](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) using a relatively small dataset of mixed manually annotated and semi-automatically annotated images that were collected from the both the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases) and the ros bags provided for training.

The images manually annotated were labelled with [LabelImg](https://github.com/tzutalin/labelImg) while for the semi-automatic annotation a small [utility](./label_data.py) was created that runs one of the pretrained models on a set of images capturing the bounding boxes and labelling them with a predefined label:

```sh
$ python label_data.py --data_dir=data\simulator\red --label=red --model_path=models\ssd_inception_v2_coco_2018_01_28\frozen_inference_graph.pb
```

Note that all the annotations in the dataset were manually verified and/or adjusted.

In order to train the model using the object detection API the images needs to be fed as a [TensorFlow Record](https://www.tensorflow.org/tutorials/load_data/tf-records), the repository contains a small [utility](./create_tf_record.py) (loosely based on the [TensorFlow object detection api](https://github.com/tensorflow/models/tree/master/research/object_detection) tool) that converts the annotated images into a [TensorFlow Record](https://www.tensorflow.org/tutorials/load_data/tf-records) optionally splitting the dataset into train and validation:

```sh
$ python create_tf_record.py --data_dir=data\simulator --labels_dir=data\simulator\labels --labels_map_path=config\labels_map.pbtxt --output_path=data\simulator\simulator.record
```

For more details about the conversion to TF Records see https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md.

Training
---

In order to train the model we use the [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection), in the following the steps that are needed to fine-tune an trained model:

1. Download the trained models from the [TensorFlow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) with their associated [pipeline configuration](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs).

2. Download the [TensorFlow Object Detection API](https://github.com/tensorflow/models/blob/master/research/object_detection) and perform the required [installation steps](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md):
   * ```git clone https://github.com/tensorflow/models.git```
   * Copy the ```research\object_detection``` and ```research\slim``` folders to the root folder
   * Get a copy of the [Protobuff Compiler](https://github.com/protocolbuffers/protobuf/) (e.g. For [Windows](https://github.com/protocolbuffers/protobuf/releases/download/v3.4.0/protoc-3.4.0-win32.zip))
   * Compile the protcol buffers: ```protoc.exe object_detection/protos/*.proto --python_out=.```
   * Set your PYTHONPATH: ```SET PYTHONPATH=%cd%;%cd%\slim``` (windows) or ```export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim``` (linux) 
   * Test that the API is installed: ```python object_detection/builders/model_builder_test.py```

3. [Configure the pipeline](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md), copies of the configurations used can be found in the [config](./config) folder.

4. Install the [COCO Api](https://github.com/cocodataset/cocoapi) 
    
    * If you run into ["ImportError: No module named 'pycocotools'"](https://github.com/matterport/Mask_RCNN/issues/6) under windows:

        ```
        pip install Cython
        pip install git+https://github.com/philferriere/cocoapi.git#egg=pycocotools^&subdirectory=PythonAPI
        ```

    * If you run into ["TypeError: can't pickle dict_values objects"](https://github.com/tensorflow/models/issues/4780) look into object_detection\model_lib.py for ```category_index.values()``` and replace with ```list(category_index.values())```


5. [Run](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_locally.md) the training session:

    ```
    python object_detection\model_main.py --pipeline_config_path=path/to/the/model/config --model_dir=path/to/the/output
    ```

6. Watch it happen with tensorboard:

        tensorboard --logdir=path/to/the/output


    And open the browser to `http://{machine_ip}:6006`

### Configuration Files

The repository contains various configuration files for the different datasets (mixed is the simulator + carla dataset) for different models, the things that I changed from the samples provided by tensorflow are:

* The various training and validation paths of the tf records (e.g. input_path elements)
* The label_map_path
* The num_classes
* The num_steps
* The num_examples in the evaluation section that correspond to the number of samples in the evaluation record
* The ssd_anchor_generator section, updating the scales and removing unused aspect ratios (the traffic lights are more or less 0.33)

### AWS

To train on AWS I used the Amazon Deep Learning AMI (v20 with tensorflow 1.12) and GPU graphics g3s.xlarge instance type (it has a more recent GPU and costs less than other GPU instances even though less ram), alternatively the GPU Compute p2.xlarge works fine (it's a tiny bit more expensive). I used spot instances with 5-6 hours request length (making sure to uncheck the delete volume option).

Once the instance is up and running we need to prepare the environment:

1. Connect to the instance:
   ```sh
   $ ssh ubuntu@instance-public-dns
   ```
2. Activate the tensorflow environment:
   ```sh
   $ source activate tensorflow_p36
   ```
3. Install the object detection API (E.g. From the linux [installation steps](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)):
   * Get the object detection API:
       ```sh
       git clone https://github.com/tensorflow/models.git tmp
       cp -r tmp/research/object_detection object_detection/
       cp -r tmp/research/slim slim/
       ```
   * Install dependencies:
        ```sh
        sudo apt-get install protobuf-compiler python-pil python-lxml python-tk
        pip install --user Cython
        pip install --user contextlib2
        pip install --user jupyter
        pip install --user matplotlib
        ```
    * Install the coco API:
        ```sh
        git clone https://github.com/cocodataset/cocoapi.git
        cd cocoapi/PythonAPI
        make
        cp -r pycocotools ../../pycocotools
        ``` 
    * Compile the proto buffers:
        ```sh
        protoc object_detection/protos/*.proto --python_out=.
        ``` 
    * Add the library to PYTHONPATH (Note: this expires with the session, put it in a script):
        ```sh
        export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
        ````
    * Test the installation:
        ```sh
        python object_detection/builders/model_builder_test.py
        ```
4. Download a model, for example SSD with Inception:
    ```sh
    mkdir models
    cd models
    wget http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2018_01_28.tar.gz
    tar -xzvf ssd_inception_v2_coco_2018_01_28.tar.gz
    cd ..
    ```
5. Download the dataset from https://drive.google.com/open?id=1NXqHTnjVC1tPjAB5DajGc30uWk5VPy7C and upload the record files to the `data` folder
6. Run the training:
    ```sh
    python object_detection/model_main.py --pipeline_config_path=config/ssd_inception_v2_coco_mixed.config --model_dir=models/fine_tuned/ssd_inception
    ```
    
    If you want to run in it background:
    
    ```
    nohup python -u object_detection/model_main.py --pipeline_config_path=config/ssd_inception_v2_coco_mixed.config --model_dir=models/fine_tuned/ssd_inception > training.log &
    ```
7. Run tensorboard:
   ```sh
   tensorboard --logdir=models/fine_tuned
   ```

   or in background:

   ```sh
   nohup tensorboard --logdir=models/fine_tuned > tensorboard.log &
   ```

NOTE: If you want to see some logging in the std out just add `tf.logging.set_verbosity(tf.logging.INFO)` after the imports in [./object_detection/model_main.py](./object_detection/model_main.py)

If your spot instance is stopped while training and you made sure to uncheck the "delete volume" option when requesting the spot instance, your volume will be retained and you can continue the training from a previous checkpoint:

1. Request a new instance
2. Go to the volumes and attach the previous volume to the new instance
3. Connect to the instance and mount the previous volume:

    ```sh
    mkdir /prev_volume
    sudo mount /dev/xvdf1 /prev_volume
    ```

    Note that the device name `xvdf1` can be found running `lsblk`.
4. Copy the old model to the new instance
   
   ```sh
    cp -r /prev_volume/home/ubuntu/models/fine_tuned /models/fine_tuned
    ```

5. Run the training with the same configuration (it will pick up the last checkpoint)

   ```sh
   python object_detection/model_main.py --pipeline_config_path=config/ssd_inception_v2_coco_mixed.config --model_dir=models/fine_tuned/ssd_inception
   ```
