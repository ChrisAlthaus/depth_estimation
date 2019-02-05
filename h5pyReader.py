import h5py
import matplotlib.pyplot as plt 

"""
Parsing the dataset structure.

"""
f = h5py.File('C:/Program Files/TensorFlow/datasets/nyu_depth_v2_labeled.mat')

keys = list(f.keys())
print("Keys= ",keys)
print(f)

refs = f['#refs#']
subsystem = f['#subsystem#']

print("refs:", refs)
print("subsystem:", subsystem.keys())

labels = f['labels']
print(labels)

plt.figure("Labels")

for i in range(10):
	label = labels[i].transpose((1,0))
	
	plt.subplot(3, 4, i+1)
	plt.imshow(label)
	
	plt.axis("off")
	plt.colorbar()
	plt.subplots_adjust(wspace=0.5)

#plt.show()


depths = f['depths']
print(depths.shape)

plt.figure("Depths")

for i in range(10):
	depthsImage = depths[i].transpose((1,0))
	
	plt.subplot(3, 4, i+1)
	plt.imshow(depthsImage)
	
	plt.axis("off")
	plt.colorbar()
	plt.subplots_adjust(wspace=0.5)



rawDepths = f['rawDepths']
print(rawDepths.shape)

plt.figure("Raw Depths")

for i in range(10):
	rawDepthsImage = rawDepths[i].transpose((1,0))
	
	plt.subplot(3, 4, i+1)
	plt.imshow(rawDepthsImage)
	
	plt.axis("off")
	plt.colorbar()
	plt.subplots_adjust(wspace=0.5)

#plt.show()

#exit(1)

images = f['images'][()]


print(images.shape)
plt.figure("Test Images")

for i in range(10):
	testImage = images[i].transpose((2,1,0))
	
	plt.subplot(3, 4, i+1)
	plt.axis("off")
	plt.imshow(testImage)
	plt.subplots_adjust(wspace=0.5)

plt.show()