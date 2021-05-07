# that-nlg
Dependencies: TF>2.1.2, numpy

nlg-train is for training models: message me if you're interested in running it. that_text_gen.py contains one class, model, which implements loading a pretrained 
and generating text from it. The typical usage is

from that_text_gen import model as generate_model

model = generate_model('marx')

model.load_model()

generated_text = model.generate_text()

where 'marx' can be any supported philosopher name. Currently accepted names are 'marx' and 'wittgenstein', but you can check the subdirectory of final_weights as the
directory names there are the valid philosopher names.

If you run into UnknownError: Fail to find the dnn implementation. [Op:CudnnRNN] try

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
