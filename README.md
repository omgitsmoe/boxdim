# Contents of this repository

This repo contains various command-line tools that were used during the writing
of my master thesis on the subject of the potential application of the boxcounting
dimension as a vitality measure for trees.

All of these use various parts of the excellent Point Cloud Library:

Rusu, R.B., & Cousins, S., 2011. 3D is here: Point Cloud Library (PCL), in: IEEE
International Conference on Robotics and Automation (ICRA). Shanghai, China.

## Computing the boxcounting dimension from a point cloud

Usage:

```
boxdim.exe minimum_edge_length algorithm point_cloud_file
```

The `minimum_edge_length` corresponds to the cutoff point at which you want
to stop the boxcounting based on the expected point cloud resolution
(the unit of the edge length depens on your point cloud data).
Usually somewhere between 5--10&nbsp;cm.

Available algorithms:

- "seidel", which is based on:
  
  Sarkar, N., & Chaudhuri, B., 1994. An efficient differential box-counting approach
  to compute fractal dimension of image. IEEE Transactions on Systems Man and
  Cybernetics 24, 115–120. https://doi.org/10.1109/21.259692
  
  Seidel, D., 2018. A holistic approach to determine tree structural complexity based
  on laser scanning data and fractal analysis. Ecology and Evolution 8, 128–134.
  https://doi.org/10.1002/ece3.3661

  - same algorithm, but using wide registers or the GPU:
    - "seidel\_sse" using SEE instructions (fastest in benchmarks, roughly 1 second for a point
      cloud with 3.7 million points)
    - "seidel\_avx" using AVX instructions
    - "seidel\_gpu" using OpenCL to use the GPU (copying the memory to the GPU is too costly
      to compete with SSE)
- "cc", based on [CloudCompare's](https://www.cloudcompare.org/) `CCLib/CCMiscTools.cpp`
- "pcl", using `pcl::octree::OctreePointCloudOccupancy`

Supports [PCD](https://pointclouds.org/documentation/tutorials/pcd_file_format.html)
files as well as ASCII text files.
Text files should not contain a header line and each point is represented by one line with
the x, y and z coordinates separated by spaces. Trailing data on the line (up to 255 chars) is ignored.

## Computing the competition index KKL from a point cloud

Usage:

```
compindex.exe voxelEdgeLength methodName coneTipHeight plotCloudFileName treeCloudFileName
```

Arguments:

- voxelEdgeLength: minimum voxel edge length that is used for the voxel grid subsampling;
  usually 10&nbsp;cm, so you would pass it as 0.1 if your point cloud has the unit meters
- methodName: eiter "cone" or "cylinder"
- coneTipHeight: height in relation to the total tree height, where the cone tip will
  be placed at, 0.2 for 20&nbsp;% of the total tree height
- plotCloudFileName: name of the PCD or ASCII file
  (see [above](#computing-the-boxcounting-dimension-from-a-point-cloud)) containing
  the whole plot or rather the trees sorrounding the subject in question
- treeCloudFileName: name of the PCD or ASCII file
  (see [above](#computing-the-boxcounting-dimension-from-a-point-cloud)) containing
  **ONLY** the tree that the KKL should be computed for

References:

Metz, J., Seidel, D., Schall, P., Scheffer, D., Schulze, E.-D., & Ammer, C., 2013.
Crown modeling by terrestrial laser scanning as an approach to assess the effect
of aboveground intra- and interspecific competition on tree growth. Forest
Ecology and Management 310, 275–288. https://doi.org/10.1016/j.foreco.2013.
08.014

Seidel, D., Hoffmann, N., Ehbrecht, M., Juchheim, J., & Ammer, C., 2015. How
neighborhood affects tree diameter increment – New insights from terrestrial
laser scanning and some methodical considerations. Forest Ecology and
Management 336, 119–128. https://doi.org/10.1016/j.foreco.2014.10.020

## Diffing point clouds

Usage:

```
diff.exe pointCloudA pointCloudB
```

Will write all the points that are only found in B into a "\_diff" file.
Only supports PCD files.

## Show various stats about a point cloud

Usage:

```
pcstats <input_point_cloud_path> <K nearest neighbours> <top % of tree crown>
```

Arguments:

- K neares neighbours: How many neighbours should be searched for
  when calculating average/max nearest neighbour distance etc.
- top % of tree crown: Which part in percent from the top of the tree
  should be used for calculating the same statistics as above

Prints out the average distance, average max distance and the overall maximum
for each 2&nbsp;m (the point cloud is assumed to be in units of meters).
The same stats are emitted for the passed top % of the tree crown.

## Euclidean clustering and region segmentation

Produces four point cloud files:

- downsampled
- euclidean-clusters (downsampled)
- \_ec\_region-based-seg (downsampled)
- \_ec\_region-based-seg\_orig (regions transferred to original point cloud based on a
  nearest neighbour search)

Usage:

```
region_seg.exe edgeLength pointCloudFile smoothness
```

Arguments:

- edgeLength: edge lengths used for the voxel grid subsampling
  (each voxel with the passed in edge length is reduced to their centroid)
- pointCloudFile: name of the PCD file
- smoothness: smoothness used for the region growing

References:

Burt, A., Disney, M., & Calders, K., 2018. Extracting individual trees from lidar
point clouds using treeseg. Methods in Ecology and Evolution. https://doi.org/
10.1111/2041-210x.13121

