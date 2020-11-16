#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <limits>
#include <chrono>  // for function timer

#include <Eigen/Core>

#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/octree/octree_pointcloud_occupancy.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>

#include "treeseg.h"

inline bool file_exists(const std::string& name) {
	if (FILE* file = fopen(name.c_str(), "r")) {
		fclose(file);
		return true;
	}
	else {
		return false;
	}
}

template<typename T>
void print_vec(const std::vector<T>& vec)
{
	std::cout << "[";
	for (int i = 0; i < vec.size(); ++i)
	{
		std::cout << vec[i] << (i != (vec.size() - 1) ? ", " : "");
	}
	std::cout << "]" << std::endl;
}

void read_input(std::string filename, pcl::PointCloud<pcl::PointXYZ>::Ptr out)
{
	if (boost::ends_with(filename, ".pcd"))
	{
		std::cout << "Reading from pcd file: " << filename << std::endl;
		pcl::PCDReader reader;
		reader.read(filename, *out);
	}
	else
	{
		std::cout << "Reading from (assumed) ascii file: " << filename << std::endl;
		std::ifstream infile(filename);
		float x, y, z;
		while (infile >> x >> y >> z)
		{
			pcl::PointXYZ point;
			point.x = x;
			point.y = y;
			point.z = z;
			out->insert(out->end(), point);
			// ignore rgb channel values, max 255 chars are ignored or till we find a \n
			infile.ignore(255, '\n');
		}
		infile.close();
	}
}

void write_result(size_t voxelsRequired, const std::string& out_filename)
{
	// standard RFC 4180 uses comma as separator
	std::ofstream outfile(out_filename);
	outfile << voxelsRequired << std::endl;
	outfile.close();
}

// remove all points in a that are also in b -> a - b
void diff(pcl::PointCloud<pcl::PointXYZ>::Ptr a, const pcl::PointCloud<pcl::PointXYZ>::Ptr b)
{
	// needed for sorting in set
	auto ptxyzLess = [](const pcl::PointXYZ& lhs, const pcl::PointXYZ& rhs)
	{
		// return (x < pt.x) || ((pt.x < x)) && (y < pt.y)) || ((!(pt.x < x)) && (!(pt.y < y)) && (z < pt.z));
		return (lhs.x < rhs.x) || ((lhs.x == rhs.x) && (lhs.y < rhs.y)) || ((lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z < rhs.z));
	};

	// type of comparator needed, so use decltype to get type of lambda
	// comparator function also needs to be passed to constructor!
	std::set<pcl::PointXYZ, decltype(ptxyzLess)> pointsB(ptxyzLess);
	for (auto pt : *b)
		pointsB.insert(pt);

	pcl::PointIndices::Ptr duplicates(new pcl::PointIndices());
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	for (size_t i = 0; i < a->size(); ++i)
	{
		if (pointsB.count(a->at(i)) > 0)
			duplicates->indices.push_back(i);
	}
	extract.setInputCloud(a);
	extract.setIndices(duplicates);
	// keep all the points but the indices in duplicates
	extract.setNegative(true);
	extract.filter(*a);
}

// copied and adapted from treeseg: https://github.com/apburt/treeseg
void find_stem_base_point(pcl::PointCloud<pcl::PointXYZ>::Ptr tree, Eigen::Vector3f* stemBasePointOut)
{
	std::cout << "Finding stem base point: Assuming stem is between -3m and 3m plot height!" << std::endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr treeZSlice(new pcl::PointCloud<pcl::PointXYZ>);
	spatial1DFilter(tree, "z", -3.0f, 3.0f, treeZSlice);

	//
	// Find and create trunk region
	//
	std::cout << "Euclidean clustering: " << std::flush;
	std::vector<std::vector<float>> nndata;
	nndata = dNNz(treeZSlice, 9, 2);
	float nnmin = std::numeric_limits<int>().max();
	float nnmax = 0;
	for (int i = 0; i < nndata.size(); i++)
	{
		if(nndata[i][1] < nnmin) nnmin = nndata[i][1];
		if(nndata[i][1] > nnmax) nnmax = nndata[i][1];
	}
	float dmax = (nnmax + nnmin) / 2;
	std::cout << dmax << " dmax" << std::endl;
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
	euclideanClustering(treeZSlice, dmax, 3, clusters);
	//
	std::cout << "Region-based segmentation: " << std::flush; 
	//int idx = findClosestIdx(tree, clusters, true);
	int idx = findPrincipalCloudIdx(clusters);
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> regions;
	int nnearest = 50;
	int nmin = 3;
	float smoothness = 12.5f;
	regionSegmentation(clusters[idx], nnearest, nmin, smoothness, regions);
	std::cout << "Done!" << std::endl;

	// assume cloud with most points is cloud containing the trunk
	auto trunkCloud = regions[0];
	for (int i = 0; i < regions.size(); ++i)
	{
		if (regions[i]->size() > trunkCloud->size()) trunkCloud = regions[i];
	}
	Eigen::Vector4f min, max;
	pcl::getMinMax3D(*trunkCloud, min, max);

	//
	//
	pcl::PointCloud<pcl::PointXYZ>::Ptr unbiasedStem(new pcl::PointCloud<pcl::PointXYZ>);
	// get slice around DBH 1.3m that's unbiased form roots or low branches
	// needs to be a longer slice otherwise the stem won't have max cluster height
	// NOTE: most of the time this slice of the trunkCloud region will already only
	//	     contain just the stem, but sometimes when dealing with free-standing
	//		 trees that have very low-hanging branches these might be part of
	//       the trunkCloud region -> so the following steps are just to be safe!
	// NOTE: change lmin when changing height of zslice!!!
	spatial1DFilter(trunkCloud /*tree*/, "z", min[2] + 1.0f, min[2] + 1.5f, unbiasedStem);

	pcl::PCDWriter writer;
	writer.write("test_unbiased.pcd", *unbiasedStem, true);

	// Find regions for RANSAC Cylinder fitting
	clusters.clear();
	std::vector<float> stSliceDNN = dNN(unbiasedStem, 10);
	euclideanClustering(unbiasedStem, stSliceDNN[0], 10, clusters);

	writeClouds(clusters, "test_clust.pcd", false);
	//
	// select trunk based on max height and min dist from avg centroid of all clusters
	//
	std::vector<Eigen::Vector4f> centroids;
	Eigen::Vector4f avgCentroid;
	Eigen::Vector4f currentCentroid;
	for (auto cl : clusters)
	{
		pcl::compute3DCentroid(*cl, currentCentroid);
		avgCentroid += currentCentroid;
		centroids.push_back(currentCentroid);
	}
	avgCentroid /= clusters.size();
	float minSqDist = std::numeric_limits<float>::max();
	float maxHeight = 0;
	auto probTrunkCluster = clusters[0];
	pcl::PointCloud<pcl::PointXYZ>::Ptr catClusters(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < clusters.size(); i++) *catClusters += *clusters[i];
	Eigen::Vector4f catMin, catMax;
	pcl::getMinMax3D(*catClusters, catMin, catMax);
	for (int i = 0; i < clusters.size(); ++i)
	{
		Eigen::Vector4f cmin, cmax;
		pcl::getMinMax3D(*clusters[i], cmin, cmax);
		// cluster has no point in bottom 20cm range of all clusters minz
		// probably redundant since we select the cluster base on max height
		if (!(cmin[2] < (min[2] + 0.2f)))
			continue;
		// when using a larger (>=.5m ?) zslice the stem cluster should have the largest height
		float height = cmax[2] - cmin[2];
		Eigen::Vector4f vecToAvgCentroid = avgCentroid - centroids[i];
		// only second criterium, since on very one-sided tree the trunk might not be closer to the
		// centroid than some parts of the crown
		float sqDist = vecToAvgCentroid.dot(vecToAvgCentroid);
		// point count was failing as criterium for selecting trunk cluster
		if ((height < maxHeight) && (sqDist < minSqDist) /*&& (clusters[i]->size() > probTrunkCluster->size())*/)
		{
			probTrunkCluster = clusters[i];
			minSqDist = sqDist;
			maxHeight = height;
		}
	}
	// idx = findPrincipalCloudIdx(clusters);
	regions.clear();
	regionSegmentation(probTrunkCluster, nnearest, nmin, smoothness, regions);

	writeClouds(regions, "test_regs.pcd", false);

	// Fit cylinder to get the radius for extracting the stem volume later
	constexpr float diameterMin = 0.5f;
	constexpr float diameterMax = 2.0f;
	std::cout << "RANSAC cylinder fits: " << std::flush;
	std::vector<cylinder> cyls;
	nnearest = 60;
	float lmin = 0.35f; // 35cm min at 50cm slice
	float stepcovmax = 0.3; //.2
	float radratiomin = 0.8; //.8
	for (int i = 0; i < regions.size(); i++)
	{
		cylinder cyl;
		fitCylinder(regions[i], nnearest, true, true, cyl);
		if (cyl.ismodel == true)
		{
			if (cyl.rad * 2 >= diameterMin && cyl.rad * 2 <= diameterMax && cyl.len >= lmin)
			{
				//std::cout << "DIM OK" << std::endl;
				//std::cout << "cov " << cyl.stepcov << " radrat " << cyl.radratio << std::endl;
				if (cyl.stepcov <= stepcovmax && cyl.radratio > radratiomin)
				{
					cyls.push_back(cyl);
				}
			}
		}
	}
	std::cout << "Done!" << std::endl;

	char fn[20];
	int n = 0;
	for (auto c : cyls)
	{
		if (c.inliers->size() == 0)
			continue;
		++n;
		sprintf(fn, "test_cyls_%d.pcd", n);
		writer.write(fn, *c.inliers, true);
	}
	if (cyls.size() != 1)
	{
		std::cout << "ERROR: Found " << (cyls.size() <= 0 ? "less" : "more") << " than one cylinder in stem slice!" << std::endl;
		exit(1);
	}

	// fitted cylinder inliers often have one side missing so we can't use it
	// for finding x/y centers
	// auto stemRing = cyls[0].inliers;
	// pcl::getMinMax3D(*stemRing, min, max);

	// use radius of fitted cylinder to get stem volume without connections to branches
	// or most of the wider bottom parts of the stem where it's connected to the roots or
	// since we fitted the cylinder at roughly breast height we enlarge the cylinder by 20%
	pcl::PointCloud<pcl::PointXYZ>::Ptr volume(new pcl::PointCloud<pcl::PointXYZ>);
	float expansionfactor = 1.2f;
	cyls[0].rad = cyls[0].rad * expansionfactor;
	spatial3DCylinderFilter(trunkCloud, cyls[0], volume);

	writer.write("test_vol.pcd", *volume, true);

	pcl::getMinMax3D(*volume, min, max);
	// get middle between min and max coordinates of stem slice
	float centerX = (min[0] + max[0]) * 0.5f;
	float centerY = (min[1] + max[1]) * 0.5f;

	stemBasePointOut->x() = centerX;
	stemBasePointOut->y() = centerY;
	// use minimum z of stem volume (root and branch connections trimmed off) as stem base point z
	stemBasePointOut->z() = min[2];

	std::cout << "Stem base point:" << std::endl << *stemBasePointOut << std::endl;
}

// plot_cloud is assumed to have had the tree removed previously
void kkl(pcl::PointCloud<pcl::PointXYZ>::Ptr plot_cloud,
		 pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud,
		 const std::string base_name, float voxelEdgeLength,
		 const Eigen::Vector3f& stemBasePoint)
{

	pcl::PointCloud<pcl::PointXYZ> filtered;
	// TODO: use voxel center instead of centroid here?
	pcl::VoxelGrid<pcl::PointXYZ> downsample;
	downsample.setInputCloud(plot_cloud);
	downsample.setLeafSize(voxelEdgeLength, voxelEdgeLength, voxelEdgeLength);
	downsample.filter(filtered);

	Eigen::Vector4f min, max;
	pcl::getMinMax3D(*tree_cloud, min, max);
	float treeHeight = max[2] - min[2];
	std::cout << "Tree height: " << treeHeight << std::endl;

	// put a search cone at 60% of total tree height over the stem base point
	// with an opening angle of 60° -> equilateral cone -> rotated equilateral triangle around cone axis
	// every tree that has a voxel falling into that cone is a competitor
	size_t nVoxelsInCone = 0;
	// normalized axis vector pointing from tip to base
	Eigen::Vector3f coneAxis(0, 0, 1);
	Eigen::Vector3f coneTip(stemBasePoint);
	// previous studies -> search cone at 60% of total tree height
	// Seidel et al. 2015: search cones fixed at a low height(1.3 m above ground for D i and 20 %
	// of TTH for D i 60 %) are suitable to describe the competitive pressure
	// by the surrounding trees based on voxel counts
	coneTip[2] += 0.6f * treeHeight;
	std::cout << "Cone tip: " << std::endl << coneTip << std::endl;

	constexpr float halfOpeningAngle = 60.0f * 0.5f;

	pcl::PointIndices::Ptr incone(new pcl::PointIndices());
	pcl::ExtractIndices<pcl::PointXYZ> extract;

	// TIMER START
	auto start = std::chrono::high_resolution_clock::now();

	//for (pcl::PointXYZ pt : filtered)
	for (size_t i = 0; i < filtered.size(); ++i)
	{
		pcl::PointXYZ pt = filtered[i];
		if (pt.z < coneTip[2])
			continue;
		Eigen::Vector3f pointVec(pt.x, pt.y, pt.z);
		// pointVec - coneTip gives us vec from coneTip to point
		Eigen::Vector3f tipToPoint = pointVec - coneTip;
#if 0
		// project that onto cone axis by computing the dot product (coneAxis needs to be normalized) -> height
		// since coneAxis is just pointing in positive Z direction it just extracts the Z component of the vector
		// float coneHeightAtPoint = tipToPoint.dot(coneAxis);
		// float coneHeightAtPoint = tipToPoint.z;
		// tan: Returns the tangent of an angle of x radians.
		// r = tan(phi) * h
		// phi = 0.5f * opening angle
		float radiusAtHeight = tan(30.0f * M_PI / 180.0f) * tipToPoint[2];
		// point's orthognal dist from cone axis
		// use squared distance and radius to avoid computation of sqrt
		// coneAxis * h = mid-point of cone at point's height
		// pointVec - midPoint = vec from mid-point to pointVec
		// dot product with itself = squared length
		float squaredOrthDist = (pointVec - tipToPoint[2] * coneAxis).squaredNorm();

		if (squaredOrthDist <= (radiusAtHeight * radiusAtHeight))
		{
			++nVoxelsInCone;
			incone->indices.push_back(i);
		}
#else
		// use angle between cone axis vector and vector from cone tip to testing point
		// to decide wheter the point is inside the search cone
		// angle <= 30 deg means it's inside the cone
		float numerator = tipToPoint.dot(coneAxis);
		// length squared to avoid sqrt -> so numerator also has to be squared
		float denom = tipToPoint.dot(tipToPoint);
		float angle = acos((numerator * numerator) / denom) * 180.0f / M_PI;

		if (angle <= halfOpeningAngle)
		{
			++nVoxelsInCone;
			incone->indices.push_back(i);
		}
#endif
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr bshared(&filtered);
	extract.setInputCloud(bshared);
	extract.setIndices(incone);
	// keep only points in incone indices
	extract.setNegative(false);
	extract.filter(filtered);

	if (filtered.size() != 0)
	{
		pcl::PCDWriter writer;
		writer.write(base_name + "_cone-inliers.pcd", filtered, true);
	}

	// TIMER END
	auto stop = std::chrono::high_resolution_clock::now();

	// Subtract stop and start timepoints and 
	// cast it to required unit. Predefined units 
	// are nanoseconds, microseconds, milliseconds, 
	// seconds, minutes, hours. Use duration_cast() 
	// function. 
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	// To get the value of duration use the count() 
	// member function on the duration object 
	std::cout << "NORMAL TIMING: " << duration.count() << " milliseconds" << std::endl;
	//
	std::cout << "Number of voxels in search cone: " << nVoxelsInCone << std::endl;

	std::string out_filename = base_name + "_kkl.txt";
	std::cout << "Writing output file: " << out_filename << std::endl;
	write_result(nVoxelsInCone, out_filename);
}

// plot_cloud is assumed to have had the tree removed previously
void kkl_cylinder(pcl::PointCloud<pcl::PointXYZ>::Ptr plot_cloud,
				  pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud,
				  const std::string base_name, float voxelEdgeLength,
				  const Eigen::Vector3f& stemBasePoint)
{

	pcl::PointCloud<pcl::PointXYZ> filtered;
	// TODO: use voxel center instead of centroid here?
	pcl::VoxelGrid<pcl::PointXYZ> downsample;
	downsample.setInputCloud(plot_cloud);
	downsample.setLeafSize(voxelEdgeLength, voxelEdgeLength, voxelEdgeLength);
	downsample.filter(filtered);

	Eigen::Vector4f min, max;
	pcl::getMinMax3D(*tree_cloud, min, max);
	float treeHeight = max[2] - min[2];
	std::cout << "Tree height: " << treeHeight << std::endl;

	size_t nVoxelsInCylinder = 0;
	Eigen::Vector3f cylMidBase(stemBasePoint);
	// Seidel et al. 2015 mentions 4m radius search cylinder performing the best
	// but no height is mentioned? setting height to stem base point height
	// will result in ground voxels being counted
	cylMidBase[2] += 2.0f;
	std::cout << "Cylinder base mid-point: " << std::endl << cylMidBase << std::endl;

	pcl::PointIndices::Ptr incyl(new pcl::PointIndices());
	pcl::ExtractIndices<pcl::PointXYZ> extract;

	// TIMER START
	auto start = std::chrono::high_resolution_clock::now();

	constexpr float radiusSquared = 4.0f * 4.0f;
	//for (pcl::PointXYZ pt : filtered)
	for (size_t i = 0; i < filtered.size(); ++i)
	{
		if (filtered[i].z < cylMidBase[2])
			continue;
		Eigen::Vector3f pointVec(filtered[i].x, filtered[i].y, filtered[i].z);
		// pointVec - coneTip gives us vec from coneTip to point
		Eigen::Vector3f midBaseToPoint = pointVec - cylMidBase;
		// normally midBaseToPoint * {0, 0, 1} but this just extracts the z component -> midBaseToPoint[2]
		float zDist = midBaseToPoint[2];
		float distSquared = midBaseToPoint.dot(midBaseToPoint) - zDist;

		if (distSquared <= radiusSquared)
		{
			++nVoxelsInCylinder;
			incyl->indices.push_back(i);
		}
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr bshared(&filtered);
	extract.setInputCloud(bshared);
	extract.setIndices(incyl);
	// keep only points in incone indices
	extract.setNegative(false);
	extract.filter(filtered);

	if (filtered.size() != 0)
	{
		pcl::PCDWriter writer;
		writer.write(base_name + "_cyl-inliers.pcd", filtered, true);
	}

	// TIMER END
	auto stop = std::chrono::high_resolution_clock::now();

	// Subtract stop and start timepoints and 
	// cast it to required unit. Predefined units 
	// are nanoseconds, microseconds, milliseconds, 
	// seconds, minutes, hours. Use duration_cast() 
	// function. 
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

	// To get the value of duration use the count() 
	// member function on the duration object 
	std::cout << "NORMAL TIMING: " << duration.count() << " milliseconds" << std::endl;
	//
	std::cout << "Number of voxels in search cylinder: " << nVoxelsInCylinder << std::endl;

	std::string out_filename = base_name + "_kkl.txt";
	std::cout << "Writing output file: " << out_filename << std::endl;
	write_result(nVoxelsInCylinder, out_filename);
}

// argv: 0: -, 1: minEdgeLength, 2: method (seidel, cc, pcl) 3: path to input file either ascii or pcd
int main(int argc, char* argv[])
{
	if (argc < 5)
	{

		std::cout << "Usage: compindex voxelEdgeLength methodName plotCloudFileName treeCloudFileName" << std::endl;
		std::cout << "       methods: seidel, cc, pcl" << std::endl;
		return 1;
	}
	float minEdgeLength = atof(argv[1]);
	if (minEdgeLength <= 0)
	{
		std::cout << "Minimal edge length needs to be > 0" << std::endl;
		return 1;
	}

	std::string plot_input_filename(argv[3]);
	std::string plot_base_name = plot_input_filename.substr(0, plot_input_filename.find_last_of('.'));
	std::string tree_input_filename(argv[4]);
	std::string base_name = tree_input_filename.substr(0, tree_input_filename.find_last_of('.'));
	pcl::PointCloud<pcl::PointXYZ>::Ptr plot_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr tree_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	std::cout << "Tree cloud: " << std::flush;
	read_input(tree_input_filename, tree_cloud);

	std::cout << "Plot cloud: " << std::flush;
	if (file_exists(plot_base_name + "_tree-removed.pcd"))
	{
		std::cout << "Reading plot cloud with removed tree! " << std::flush;
		read_input(plot_base_name + "_tree-removed.pcd", plot_cloud);
		std::cout << "Done!" << std::endl;
	}
	else
	{
		read_input(plot_input_filename, plot_cloud);
		std::cout << "Removing tree from plot cloud! " << std::flush;
		diff(plot_cloud, tree_cloud);
		std::cout << "Done!" << std::endl;
		pcl::PCDWriter writer;
		writer.write(plot_base_name + "_tree-removed.pcd", *plot_cloud, true);
	}

	Eigen::Vector3f stemBasePoint;
	std::string stbp_filename = base_name + "_stem-base-point.xyz";
	if (file_exists(stbp_filename))
	{
		std::cout << "Reading stem base point from file!" << std::endl;
		std::ifstream infile(stbp_filename);
		infile >> stemBasePoint[0] >> stemBasePoint[1] >> stemBasePoint[2];
		infile.close();
		std::cout << "Stem base point:" << std::endl << stemBasePoint << std::endl;
	}
	else
	{
		find_stem_base_point(tree_cloud, &stemBasePoint);
		std::ofstream outfile(stbp_filename);
		outfile << stemBasePoint[0] << " " << stemBasePoint[1] << " " << stemBasePoint[2] << std::endl;
		outfile.close();
	}


	if (strcmp(argv[2], "cone") == 0)
	{
		std::cout << "==== COMPUTING KKL USING SEARCH CONE ====" << std::endl;
		kkl(plot_cloud, tree_cloud, base_name, minEdgeLength, stemBasePoint);
	}
	else if (strcmp(argv[2], "cylinder") == 0)
	{
		std::cout << "==== COMPUTING KKL USING SEARCH CYLINDER ====" << std::endl;
		kkl_cylinder(plot_cloud, tree_cloud, base_name, minEdgeLength, stemBasePoint);
	}
	else
	{
		std::cout << "ERROR: Method not found, available methods are seidel, seidel_avx, cc, pcl" << std::endl;
	}

	return 0;
}
