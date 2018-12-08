#%% [markdown]
# # Cat - Dog classification
# ## 1. Load an image 
# 0CirmiuszRex 1 2 3

#%%
from PIL import Image
image = Image.open("./Test/0CirmiuszRex.jpg")
from IPython.display import display
display(image)

#%% [markdown]
# ## 2. Add the trained model

#%%
import tensorflow as tf
import os
import numpy as np
gd = tf.GraphDef()
gd.ParseFromString(open('model.pb', 'rb').read())
tf.import_graph_def(gd, name='')
# resize, rgb->bgr
image = image.resize((227,227))
r,g,b = np.array(image).T
image = np.array([b,g,r]).transpose()
print(image.shape)

#%% [markdown]
# ## 3. Do the prediction

#%%
sess = tf.Session()
prob_tensor = sess.graph.get_tensor_by_name('loss:0')
predictions = sess.run(prob_tensor, {'Placeholder:0': [image] })
highest_probability_index = np.argmax(predictions)
from IPython.display import Markdown
ck = ['Cat', 'Dog'][highest_probability_index]
#print(ck)
display(Markdown('# ' + ck + ''))


