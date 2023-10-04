#!/usr/bin/env python
# coding: utf-8

# ![brick1.png](attachment:brick1.png)

# In[7]:


get_ipython().system('pip install opencv-python')


# In[8]:


import cv2


# In[9]:


import numpy as np


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


image = cv2.imread('brick1.png')


# In[35]:


gray = cv2.cvtColor(cv2.imread(r"C:\Users\hp\Desktop\Dip Project\brick1.png"), cv2.COLOR_BGR2GRAY)


# In[36]:


plt.imshow(gray, cmap='gray');


# In[37]:


blur = cv2.GaussianBlur(gray, (11,11), 0)


# In[38]:


plt.imshow(blur, cmap='gray')


# In[39]:


canny = cv2.Canny(blur, 30, 150, 3)
plt.imshow(canny, cmap='gray')


# In[40]:


dilated = cv2.dilate(canny, (1,1), iterations = 2)
plt.imshow(dilated, cmap='gray')


# In[42]:


(cnt, heirarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(cv2.imread(r"C:\Users\hp\Desktop\Dip Project\brick1.png"), cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0,255,0), 2)


# In[43]:


plt.imshow(rgb)


# In[45]:


print('Bricks in the image: ', len(cnt))
plt.show()


# In[ ]:




