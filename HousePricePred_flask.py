#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pickle
from flask import Flask, render_template, request


# In[2]:


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


# In[3]:


@app.route('/')
def home():
    return render_template('index.html')


# In[4]:


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Based on data input House Price is : ${}'.format(output))


# In[ ]:


if __name__=='__main__':
    app.run(debug=True)

