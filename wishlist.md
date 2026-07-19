Create a plan for the implementation of these features. then partition the plans into sub plans that can be run concurrently. save all the plans into a folder called claudes plan. 

go through the ClaudesPlan with sonnet and compact the plan files into a code history that would be useful for implemetation of future software. 


from a point cloud of centroid loactions we would like to separate volumes of dense points from each other. the space is laid out as granules of hydrogel with beads embedded in them, clouds of markers belong to various granules in the system. We need to separate the point clouds using a k-means or gaussian mix mode distribution to determine which granule any given point belongs to. granules are roughly superellipsoids with p=4-5 and some surface roughness and elongation. the user will seed an initial value estimate of the number of granules in the volume, but this number can be relaxed by a user defined percentage. then we will use the point clouds that belong to each granule to do a tesselation to define the boundaries. tesselations that are as densily packed as their neighbors should be combined to belong to the same granule. next we should make a mask based on the approximate border of this tesselated boundary. The user will define some smoothing parameter for the volume. this continuous volume will be turned into masks for the z layer it belongs to which should be discritized by the floor of the centroid closest to the z plane from the confocal slices. 

node 1) tesselation algo
inputs: point cloud of centroidal locations
output: tesselation of the point cloud.

node 2) clustering algorithm
inputs:  point cloud, and or tesselation, other parameters we previously specify
output: labeled voxels belonging to the same 

node 3) volume mask algo
input: labeled voxels
output: masks for each volume
output: masks for all volumes

node 4) volume viewer
input: masks for volume, and or voxel metadata, smoothing parameter
output: 3D volume of the object
note: meta data attached to voxels could be: labels, displacement, strain, or any other scalar values. these scalar values will be painted ontot he surface of the volume reconstruction. 

node 5) boundary exctaction 
input: labeled voxels
output: the neighbor voxels away from the surface normal i.e. the dierction that there is not the same value. up to a number of voxels assigned by the user. 
option: do eculidian distance transform to pick voxels belonging to the boundary. 
note: this will be used to mask the volume to recover any data that is around the boundary of the volume 


DONT DO THIS NOW! wait for me to tell you when. 
go here and compare our workflows to the suggestions from matt pocock and detail how we can improve and where we do abetter job. then tell me why. 
https://github.com/mattpocock/skills/blob/main/skills