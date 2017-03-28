**Important**: Using Caffe, Tensorflow or other Deep Learning tools on Palmetto
requires you to be part of the `singularity` group on the cluster. You can check
if you are part of the `singularity` group by logging in and running the `groups`
command:

~~~
[atrikut@user001 ~]$ groups
cuuser singularity
~~~

If you don't see `singularity` in the output of the `groups` command, you are not
part of the `singularity` user group. Please contact <ithelp@clemson.edu>
and include the word "Palmetto" in the subject line with a request to be added
to the `singularity` group.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Deep Learning Tools on Palmetto Cluster](#deep-learning-tools-on-palmetto-cluster)
  - [Requesting nodes with GPUs](#requesting-nodes-with-gpus)
  - [The `deep-learning` module](#the-deep-learning-module)
  - [Tensorflow](#tensorflow)
  - [Caffe](#caffe)
  - [DIGITS](#digits)
    - [MAC OS X/Linux](#mac-os-xlinux)
      - [Starting the DIGITS server](#starting-the-digits-server)
      - [Port-forwarding from compute node to local machine](#port-forwarding-from-compute-node-to-local-machine)
    - [Windows](#windows)
  - [Tensorflow and Caffe from Jupyter Notebooks](#tensorflow-and-caffe-from-jupyter-notebooks)
    - [Configuring the Notebook Kernel](#configuring-the-notebook-kernel)
    - [Running the Notebook Server and creating/editing Notebooks:](#running-the-notebook-server-and-creatingediting-notebooks)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Deep Learning Tools on Palmetto Cluster

This document aims to explain the different
tools for Deep Learning available on the Palmetto
cluster, and how to configure and use these tools.

## Requesting nodes with GPUs

Once logged in,
to request a node with a GPU in Palmetto:

~~~
[atrikut@user001 ~]$ qsub -I -l select=1:ncpus=2:mem=12gb:ngpus=1:gpu_model=k20,walltime=72:00:00
qsub (Warning): Interactive jobs will be treated as not rerunnable
qsub: waiting for job 553382.pbs02 to start
qsub: job 553382.pbs02 ready

[atrikut@node1689 ~]$
~~~

## The `deep-learning` module

To use the various deep learning tools, add the `deep-learning` module.
Remember to purge any existing modules:

~~~
[atrikut@node1689 ~]$ module purge
[atrikut@node1689 ~]$ module add deep-learning
~~~

## Tensorflow

With the `deep-learning` module loaded,
you can import the `tensorflow` package in Python:

~~~
atrikut@node1689 ~]$ ipython
Python 2.7.6 (default, Mar 22 2014, 22:59:56)
Type "copyright", "credits" or "license" for more information.

IPython 5.3.0 -- An enhanced Interactive Python.
?         -> Introduction and overview of IPython's features.
%quickref -> Quick reference.
help      -> Python's own help system.
object?   -> Details about 'object', use 'object??' for extra details.

In [1]: import tensorflow
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcublas.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcudnn.so.5 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcufft.so.8.0 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcuda.so.1 locally
I tensorflow/stream_executor/dso_loader.cc:135] successfully opened CUDA library libcurand.so.8.0 locally

In [2]:
~~~

## Caffe

The `caffe` command-line utilities, as well
as the `caffe` Python package are also available:

~~~
atrikut@node1689 ~]$ caffe --help
caffe: command line brew
usage: caffe <command> <args>

commands:
  train           train or finetune a model
  test            score a model
  device_query    show GPU diagnostic information
  time            benchmark model execution time

  Flags from src/gflags.cc:
    -flagfile (load flags from file) type: string default: ""
    -fromenv (set flags from the environment [use 'export FLAGS_flag1=value'])
      type: string default: ""
    -tryfromenv (set flags from the environment if present) type: string
      default: ""
    -undefok (comma-separated list of flag names that it is okay to specify on
      the command line even if the program does not define a flag with that
      name.  IMPORTANT: flags in this list that have arguments MUST use the
.
.
.
~~~

This is the NVIDIA flavor of caffe (https://github.com/NVIDIA/caffe) version 0.15.14.

## DIGITS

### MAC OS X/Linux

Using DIGITS is a two-step process:

1. Starting the DIGITS server on the compute node
2. Binding a port on the local machine to the port on the compute node where the DIGITS server is running

#### Starting the DIGITS server

A DIGITS server can be started on this compute node using the following command:

~~~
atrikut@node1689 ~]$ digits-devserver
~~~

If you run into an error `Address already in use:`, try the following:

~~~
atrikut@node1689 ~]$ digits-devserver -p 5001
~~~

(or any other number between 5001-6000)

#### Port-forwarding from compute node to local machine

Open a Terminal on the local machine,
and type in the following command:

~~~
$ ssh -L 10000:node1689.palmetto.clemson.edu:5000 <username>@user.palmetto.clemson.edu
~~~

In the above command, replace

1. `node1689` with the actual compute node on which the DIGITS server was started
2. `<username`> with your Palmetto user name
3. `5000` with the port number used above. If you didn't use a different port number,
then leave as `5000`.
4. Open a web browser locally, and go to `http://localhost:10000`. You should
see the DIGITS interface.


**Note**: I've noticed that doing this a second time can cause problems sometimes.
It may help to change the local port (to e.g., 10001).

### Windows

On Windows, there are issues with the recommended SSH Client and port-forwarding,
so instead, users will start the DIGITS server on the compute node,
and also run Firefox on the compute node using X11 tunneling:

1. Download and launch the Xming X server. See instructions
[here](https://www.palmetto.clemson.edu/palmetto/beta/pages/userguide/howtos/run_graphical_applications.html).

2. Start the SSH Secure Shell Client. Go to Tools->Settings->Tunneling,
and ensure that the "Tunnel X11 Connections" box is checked.

3. Log in to Palmetto and request an interactive session on a GPU node.

4. Start the DIGITS server runnning in the background:

~~~
atrikut@node1689 ~]$ module add deep-learning
atrikut@node1689 ~]$ digits-devserver&
~~~

Then run `firefox`:

~~~
atrikut@node1689 ~]$ firefox
~~~

## Tensorflow and Caffe from Jupyter Notebooks

### Configuring the Notebook Kernel

This is a one time setup you need to do to enable using Tensorflow/Caffe
from Jupyter Notebooks on Palmetto. Simply enter the following command
(on a compute node or the user node):

~~~
$ mkdir -p ~/.local/share/jupyter/kernels
$ cp -r /software/experimental/deep-learning-kernel/kernels/deep-learning/ ~/.local/share/jupyter/kernels
~~~

That's it. You are ready to proceed to the next step:

### Running the Notebook Server and creating/editing Notebooks:

A Notebook Server can be started by logging on to `palmetto.clemson.edu/jupyterhub`.
When requesting resources for your Notebook server, ask for:

1. 1 chunk of hardware
1. 2 CPU Cores
1. 1 GPU
1. 120gb RAM (this will ensure landing on a node with a k20/k40 GPU)
1. workq queue

Once logged in to the Notebook server,
you can create a new Jupyter Notebook with support for Caffe/TensorFlow
by clicking New->"Python 2 (Deep Learning)".
For existing Notebooks, you should first open the Notebook,
and then change the kernel
by clicking on "Kernel" in the menu bar, then "Change kernel",
then "Python 2 (Deep Learning)".
