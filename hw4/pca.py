from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


temp = np.zeros((1,4096))
img_arr = np.zeros((100,64*64))
 #save 100 image in an array [100*4096]
for i in range(10):
	for j in range(10):
		file = './faceExpressionDatabase/'+chr(i+65)+'0'+str(j)+'.bmp'
		img = Image.open(file)
		width, height = img.size
		
		for y in range(height):
			for x in range(width):
				rgba = img.getpixel((x,y))
				temp[0][y*64+x] = rgba
		img_arr[i*10+j] = temp[0]

mean_face = np.mean(img_arr,axis=0)
img_mean_face = mean_face.reshape(64,64)




plt.figure(1)
plt.imshow(img_mean_face, cmap='gray')




# do SVD
X = (img_arr - mean_face)
(u,s,v) = np.linalg.svd(X.T,full_matrices=False)
"""--------
 problem 1
--------"""
# take 9 components(eigenvector)
eigenface = u[:,0:9]	


#plt.figure(num='eigenface',figsize=(3,3))
plt.figure(2)

for i in range(9):
	plt.subplot(3,3,i+1)
	plt.imshow(eigenface[:,i].reshape(64,64),cmap='gray')
	plt.axis('off')




"""---------
 problem 2
 --------"""
# plot original 100 faces
#plt.figure(num='original 100 faces',figsize)
plt.figure(3)
for i in range(100):
	plt.subplot(10,10,i+1)
	plt.imshow(img_arr[i].reshape(64,64),cmap='gray')
	plt.axis('off')

"""
plt.figure(3)
eigenface1 = u[:,0:5]
for i in range(100):
	coefficient = v[0:5,i]*s[0:5]

	img = eigenface1.dot(coefficient.T)
	plt.subplot(10,10,i+1)
	plt.imshow(img.reshape(64,64),cmap='gray')
	plt.axis('off')
"""

C = np.diag(s[0:5]).dot(v[0:5,:])
Y = u[:,0:5].dot(C)
plt.figure(4)
for i in range(100):
	plt.subplot(10,10,i+1)
	plt.imshow((Y[:,i]+mean_face).reshape(64,64),cmap='gray')
	plt.axis('off')

plt.show()

"""-----------
  problem 3
------------"""

for k in range(1,101):
	C = np.diag(s[0:k]).dot(v[0:k,:])
	X_head = u[:,0:k].dot(C)
	X_error = X - X_head.T # X_error.shape = (100,4096)
	error = 0
	for j in range(100):
		error = error + np.linalg.norm(X_error[j])**2
	error = (error/(100*4096))**0.5
	if error<2.56:
		print ('k = ',k,'error = ',error)










