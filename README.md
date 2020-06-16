# NOT WORKING YET
Currenet Environment:
absl-py==0.4.1
anytree==2.7.0
appnope==0.1.0
astor==0.7.1
backports-abc==0.4
backports.ssl-match-hostname==3.5.0.1
bleach==1.5.0
certifi==2018.4.16
chardet==3.0.4
cloudpickle==1.4.1
colorama==0.3.9
cycler==0.10.0
Cython==0.25.2
decorator==4.4.2
entrypoints==0.2.3
future==0.16.0
futures==3.1.1
gast==0.2.0
get==2018.11.19
grpcio==1.14.1
gym==0.12.1
h5py==2.10.0
html5lib==0.9999999
idna==2.7
imageio==2.4.1
imgaug==0.2.0
inflection==0.3.1
ipykernel==4.2.2
ipython==4.0.1
ipython-genutils==0.1.0
ipywidgets==7.1.1
jedi==0.11.1
Jinja2==2.8
jsonschema==2.5.1
jupyter==1.0.0
jupyter-client==4.1.1
jupyter-console==5.2.0
jupyter-core==4.0.6
Keras==2.1.6
Keras-Applications==1.0.2
Keras-Preprocessing==1.0.1
kiwisolver==1.2.0
lxml==4.2.4
Markdown==2.6.11
MarkupSafe==0.23
matplotlib==2.2.5
mistune==0.7.1
more-itertools==4.3.0
msgpack==0.5.6
nbconvert==4.1.0
nbformat==4.0.1
networkx==2.4
notebook==4.0.6
numpy==1.14.0
opencv-python==3.4.3.18
pandas==0.23.3
pandocfilters==1.4.2
parso==0.1.1
path.py==8.1.2
pickleshare==0.5
Pillow==5.1.0
post==2018.11.20
progressbar2==3.38.0
prompt-toolkit==1.0.15
protobuf==3.6.1
public==2018.11.20
pycocotools @ git+https://github.com/philferriere/cocoapi.git@2929bd2ef6b451054755dfd7ceb09278f935f7ad#subdirectory=PythonAPI
pyglet==1.3.2
Pygments==2.0.2
pyparsing==2.2.0
pyPdf==1.13
PyPDF2==1.26.0
pyreadline==2.1
python-dateutil==2.6.1
python-Levenshtein==0.12.0
python-utils==2.4.0
pytube==9.5.1
pytz==2017.3
PyWavelets==1.1.1
pywinpty==0.5.1
PyYAML==5.3.1
pyzmq==17.1.2
qtconsole==4.3.1
Quandl==3.4.1
query-string==2018.11.20
regex==2018.8.17
request==2018.11.20
requests==2.19.1
scikit-image==0.13.0
scikit-learn==0.19.2
scipy==1.1.0
selenium==3.9.0
Send2Trash==1.4.2
Shapely==1.7.0
simplegeneric==0.8.1
simplejson==3.16.0
six==1.11.0
sklearn==0.0
tensorboard==1.10.0
tensorflow==1.10.0
tensorflow-gpu==1.10.0
tensorflow-tensorboard==1.5.0
tensorlayer==1.10.1
termcolor==1.1.0
terminado==0.8.1
testpath==0.3.1
torch==1.0.1
torchvision==0.2.2.post3
tornado==4.3
tqdm==4.23.4
traitlets==4.0.0
tushare==1.2.12
urllib3==1.23
virtualenv==15.1.0
wcwidth==0.1.7
Werkzeug==0.14.1
widgetsnbextension==3.1.3
wrapt==1.10.11
xlrd==1.2.0

# Mobile Mask R-CNN
This is a Mask R-CNN implementation with MobileNet V1/V2 as Backbone architecture to be finally able to deploy it on mobile devices such as the Nvidia Jetson TX2.
The major changes to the original [matterport project](https://github.com/matterport/Mask_RCNN) are: <br />
- [X] Add Mobilenet V1 and V2 as backbone options (besides ResNet 50 and 101) + dependencies in the model
- [X] Make the whole project py2 / py3 compatible (original only works on py3)
- [X] Investigate Training Setup for Mobilenet V1 and implement it in `coco_train.py`
- [X] Add a Speedhack to mold /unmold image functions
- [X] Make the project lean and focused on COCO + direct training on passed class names (IDs before)
- [ ] Inclue more speed up options to the Model (Light-Head RCNN)
- [X] Release a trained Mobile_Mask_RCNN Model
<br />

## Getting Started
- install required packages (mostly over pip)
- clone this repository
- download and setup the COCO Dataset: `setup_coco.py`
- inside `coco.py` subclass `Config` (defined in `config.py`) and change model params to your needs
- train `mobile mask r-cnn` on COCO with: `train_coco.py`
- evaluate your trained model with: `eval_coco.py`
- do both interactively with the notebook `train_coco.ipynb`
- if you face killed kernels due to memory errors, use `bash train.sh` for infinite training
- visualize / control training with tensorboard: `cd` into your current log dir and run: <br />
`tensorboard --logdir="$(pwd)"`
- inspect your model with `notebooks/`: <br />
`inspect_data.ipynb`,`inspect_model.ipynb`, `inspect_weights.ipynb`,`detection_demo.ipynb`
- convert keras h5 to tensorflow .pb model file, in `notebooks/` run: <br />
`export_model.ipynb`
<br />


## Performance
Mobile Mask R-CNN trained on 512x512 input size
- 100 Proposals: 0.22 mAP (VOC) @ 250ms
- 1000 Proposals: 0.25 mAP (VOC) @ 330ms
<br />

## Requirements
- numpy
- scipy
- Pillow
- cython
- matplotlib
- scikit-image
- tensorflow>=1.3.0
- keras>=2.1.5
- opencv-python
- h5py
- imgaug
- IPython[all]
- pycocotools
