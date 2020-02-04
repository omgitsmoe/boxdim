#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <limits>

#include <Eigen/Core>

#include <pcl/common/common.h>
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/octree/octree_pointcloud_occupancy.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

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

void write_result(const std::vector<size_t>& voxelsRequired, const std::string& out_filename)
{
	// standard RFC 4180 uses comma as separator
	std::ofstream outfile(out_filename);
	for (int i = 0; i < voxelsRequired.size(); ++i)
	{
		// use left-shift instead of 2^i
		outfile << log(1 << i) << ", " << log((double)voxelsRequired[i]) << std::endl;
	}
	outfile.close();
}

void box_dim_seidel(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::string base_name, float minEdgeLength)
{
	Eigen::Vector4f min, max;
	// get min and max coordinates in each dimension which gives us the AABB
	pcl::getMinMax3D(*cloud, min, max);

	std::cout << "Min coordinates:\n" << min << "\nMax coordinates:\n" << max << std::endl;

	// get absolute dist between min and max coordinates to get the largest edge
	// length that we're gonna need for our octree
	Eigen::Vector4f absDist = (min - max).cwiseAbs();

	//std::cout << "Max edge length 3d:\n" << absDist << std::endl;

	// round abs dist to 2 decimal points
	float* absDat = absDist.data();  // data returns pointer to C-array
	for (int i = 0; i < 4; i++)
	{
		*absDat = roundf(*absDat * 100.0f) / 100.0f;  // use ceil to always round up so no point can be outside of our max box edge length?
		++absDat;
	}
	absDat = nullptr;

	//std::cout << "Max edge length 3d 2 decs:\n" << absDist << std::endl;

	float maxEdgeLength = absDist.maxCoeff();

	std::cout << "Max edge length: " << maxEdgeLength << std::endl;

	// The resize() method(and passing argument to constructor is equivalent to
	// that) will insert or delete appropriate number of elements to the vector
	// to make it given size(it has optional second argument to specify their
	// value).It will affect the size(), iteration will go over all those
	// elements, push_back will insert after themand you can directly access
	// them using the operator[].
	// The reserve() method only allocates memory, but leaves it
	// uninitialized.It only affects capacity(), but size() will be
	// unchanged.There is no value for the objects, because nothing is added
	// to the vector.If you then insert the elements, no reallocation will
	// happen, because it was done in advance, but that's the only effect.
	std::vector<float> edgeLengths;
	edgeLengths.reserve(20);  // prob. never going over 20 subdivisions;
	float currentEdgeLength = maxEdgeLength;
	// each octree level the edge length gets halfed, continue until we reach minEdgeLength
	// dr. seidel minEdgeLength 0.10
	// edge length at level n (starting at 0, otherwise n-1 when starting at 1) = maxEdgeLength * (1 / 2 ^ n);
	for (int i = 1; currentEdgeLength >= minEdgeLength; ++i)
	{
		edgeLengths.push_back(currentEdgeLength);
		currentEdgeLength = maxEdgeLength * 1 / (1 << i);
		//currentEdgeLength /= 2.0f;
	}

	// print edge lengths
	std::cout << "Edge lengths:" << std::endl;
	print_vec(edgeLengths);

	// relation of edge length to max edge length
	// these are just constants, since we just half the edge length each octree level
	// relative edge length on level n = 1 / 2 ^ n;
	float relEdgeLengths[] = {   1.0f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f, 0.00390625f,
								 0.001953125f, 0.0009765625f, 0.00048828125f, 0.000244140625f,
								 0.0001220703125f, 0.00006103515625f, 0.00003051757813f, 0.00001525878906f,
								 0.000007629394531f, 0.000003814697266f, 0.000001907348633f,
								 0.0000009536743164f, 0.0000004768371582f, 0.0000002384185791f,
								 0.0000001192092896f, 0.00000005960464478f, 0.00000002980232239f };
	constexpr int relEdgeLengths_len = sizeof(relEdgeLengths) / sizeof(relEdgeLengths[0]);

	// check not needed anymore since we just use 1 << i when generating log/log values
	//if (edgeLengths.size() > relEdgeLengths_len)
	//{
	//	std::cout << "ERROR: Not enough relative edge lengths (" << relEdgeLengths_len << " vs. " << edgeLengths.size() << ") available!!!" << std::endl;
	//	return 1;
	//}

	// redundant since constant 1, 1/2, 1/4, ..
	//std::vector<float> relEdgeLengths;
	//for (int i = 0; i < edgeLengths.size(); ++i)
	//{
	//	relEdgeLengths.push_back(edgeLengths[i] / maxEdgeLength);
	//}
	//std::cout << "Relative edge lengths: ";
	//for (int i = 0; i < relEdgeLengths.size(); ++i) std::cout << relEdgeLengths[i] << ", ";
	//std::cout << std::endl;

	// inverse edge lengths -> multiplying a point's coordinates gives voxel indices for that point
	std::vector<float> voxelIdxFactors;
	voxelIdxFactors.reserve(20);  // prob. never going over 20 subdivisions;
	for (int i = 0; i < edgeLengths.size(); ++i)
	{
		voxelIdxFactors.push_back(1.0f / edgeLengths[i]);
	}
	std::cout << "Voxel idx factors:" << std::endl;
	print_vec(voxelIdxFactors);

	std::vector<size_t> voxelsRequired;
	voxelsRequired.reserve(20);
	// can't store c array in STL container since it needs to be copyable and assignable
	// use struct instead, but then we need to define < opertor since std::set
	// uses std::less<T> by default; but we could also pass our own comparator function to
	// std::set<T, compFunc>
	struct voxelIdx
	{
		float x;
		float y;
		float z;
		// needed for sorting in set
		bool operator <(const voxelIdx& pt) const
		{
			// return (x < pt.x) || ((pt.x < x)) && (y < pt.y)) || ((!(pt.x < x)) && (!(pt.y < y)) && (z < pt.z));
			return (x < pt.x) || ((x == pt.x) && (y < pt.y)) || ((x == pt.x) && (y == pt.y) && (z < pt.z));
		}
	};
	voxelIdx currentVoxelIdx;
	std::set<voxelIdx> usedVoxels;
	for (int i = 0; i < edgeLengths.size(); ++i)
	{
		for (pcl::PointXYZ pt : *cloud)
		{
			// DGM: if we admit that cs >= 0, then the 'floor' operator is useless (int cast = truncation)
			//currentVoxelIdx.x = static_cast<int>(/*floor*/(pt.x - min(0)) * voxelIdxFactors[i]);
			//currentVoxelIdx.y = static_cast<int>(/*floor*/(pt.y - min(1)) * voxelIdxFactors[i]);
			//currentVoxelIdx.z = static_cast<int>(/*floor*/(pt.z - min(2)) * voxelIdxFactors[i]);
			// dr. seidel method
			currentVoxelIdx.x = roundf(pt.x * voxelIdxFactors[i] + 0.5f);
			currentVoxelIdx.y = roundf(pt.y * voxelIdxFactors[i] + 0.5f);
			currentVoxelIdx.z = roundf(pt.z * voxelIdxFactors[i] + 0.5f);
			usedVoxels.insert(currentVoxelIdx);
		}

		voxelsRequired.push_back(usedVoxels.size());
		usedVoxels.clear();
	}
	std::cout << "Number of used voxels:" << std::endl;
	print_vec(voxelsRequired);

	std::string out_filename = base_name + "_seidel.csv";
	std::cout << "Writing output csv: " << out_filename << std::endl;
	write_result(voxelsRequired, out_filename);
}

void box_dim_cc(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::string base_name, float minEdgeLength)
{
	Eigen::Vector4f min, max;
	// get min and max coordinates in each dimension which gives us the AABB
	pcl::getMinMax3D(*cloud, min, max);

	std::cout << "Min coordinates:\n" << min << "\nMax coordinates:\n" << max << std::endl;

	// get absolute dist between min and max coordinates to get the largest edge
	// length that we're gonna need for our octree
	Eigen::Vector4f absDist = (min - max).cwiseAbs();

	//std::cout << "Max edge length 3d:\n" << absDist << std::endl;

	float maxEdgeLength = absDist.maxCoeff();

	std::cout << "Max edge length: " << maxEdgeLength << std::endl;

	// from CCLib CCMiscTools.cpp und GPL v2+
	//build corresponding 'square' box
	{
		constexpr double enlargeFactor = 0.001; // CC: 0.01;
		//enlarge it if necessary
		if (enlargeFactor > 0)
			maxEdgeLength = static_cast<float>(static_cast<double>(maxEdgeLength) * (1.0 + enlargeFactor));

		// vec3 in CC
		// diagonal of the square
		Eigen::Vector4f dd(maxEdgeLength, maxEdgeLength, maxEdgeLength, 0);
		// md/2 gives us the midway point between min and max
		Eigen::Vector4f md = max + min;

		// md/2 gives us the midway point between min and max
		// -dd/2 gives us the minimum of the square with that mid point
		min = (md - dd) * 0.5f;
		// min pt + diagonal -> max point of square
		max = min + dd;
	}
	// END CC

	std::vector<float> edgeLengths;
	edgeLengths.reserve(20);  // prob. never going over 20 subdivisions;
	float currentEdgeLength = maxEdgeLength;
	// each octree level the edge length gets halfed, continue until we reach minEdgeLength
	// dr. seidel minEdgeLength 0.10
	// edge length at level n (starting at 0, otherwise n-1 when starting at 1) = maxEdgeLength * (1 / 2 ^ n);
	for (int i = 1; currentEdgeLength >= minEdgeLength; ++i)
	{
		edgeLengths.push_back(currentEdgeLength);
		currentEdgeLength = maxEdgeLength * 1 / (1 << i);
		//currentEdgeLength /= 2.0f;
	}

	// print edge lengths
	std::cout << "Edge lengths:" << std::endl;
	print_vec(edgeLengths);

	// inverse edge lengths -> multiplying a point's coordinates gives voxel indices for that point
	// mult faster than div otherwise we could just do static_cast<int>(/*floor*/(pt.x - min(0)) / edgeLengths[i])
	// when computing voxel indices
	std::vector<float> voxelIdxFactors;
	voxelIdxFactors.reserve(20);  // prob. never going over 20 subdivisions;
	for (int i = 0; i < edgeLengths.size(); ++i)
	{
		voxelIdxFactors.push_back(1.0f / edgeLengths[i]);
	}
	std::cout << "Voxel idx factors:" << std::endl;
	print_vec(voxelIdxFactors);

	std::vector<size_t> voxelsRequired;
	voxelsRequired.reserve(20);
	struct voxelIdx
	{
		float x;
		float y;
		float z;
		// needed for sorting in set
		bool operator <(const voxelIdx& pt) const
		{
			// return (x < pt.x) || ((pt.x < x)) && (y < pt.y)) || ((!(pt.x < x)) && (!(pt.y < y)) && (z < pt.z));
			return (x < pt.x) || ((x == pt.x) && (y < pt.y)) || ((x == pt.x) && (y == pt.y) && (z < pt.z));
		}
	};
	voxelIdx currentVoxelIdx;
	std::set<voxelIdx> usedVoxels;
	for (int i = 0; i < edgeLengths.size(); ++i)
	{
		for (pcl::PointXYZ pt : *cloud)
		{
			// DGM: if we admit that cs >= 0, then the 'floor' operator is useless (int cast = truncation)
			currentVoxelIdx.x = static_cast<int>(/*floor*/(pt.x - min(0)) * voxelIdxFactors[i]);
			currentVoxelIdx.y = static_cast<int>(/*floor*/(pt.y - min(1)) * voxelIdxFactors[i]);
			currentVoxelIdx.z = static_cast<int>(/*floor*/(pt.z - min(2)) * voxelIdxFactors[i]);
			usedVoxels.insert(currentVoxelIdx);
		}

		voxelsRequired.push_back(usedVoxels.size());
		usedVoxels.clear();
	}
	std::cout << "Number of used voxels:" << std::endl;
	print_vec(voxelsRequired);

	std::string out_filename = base_name + "_cc.csv";
	std::cout << "Writing output csv: " << out_filename << std::endl;
	write_result(voxelsRequired, out_filename);
}

void box_dim_pcl(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::string base_name, float minEdgeLength)
{
	Eigen::Vector4f min, max;
	// get min and max coordinates in each dimension which gives us the AABB
	pcl::getMinMax3D(*cloud, min, max);

	std::cout << "Min coordinates:\n" << min << "\nMax coordinates:\n" << max << std::endl;

	// get absolute dist between min and max coordinates to get the largest edge
	// length that we're gonna need for our octree
	Eigen::Vector4f absDist = (min - max).cwiseAbs();

	//std::cout << "Max edge length 3d:\n" << absDist << std::endl;

	// round abs dist to 2 decimal points
	float* absDat = absDist.data();  // data returns pointer to C-array
	for (int i = 0; i < 4; i++)
	{
		*absDat = ceil(*absDat * 100.0f) / 100.0f;  // use ceil to always round up so no point can be outside of our max box edge length?
		++absDat;
	}
	absDat = nullptr;

	//std::cout << "Max edge length 3d 2 decs:\n" << absDist << std::endl;

	float maxEdgeLength = absDist.maxCoeff();

	std::cout << "Max edge length: " << maxEdgeLength << std::endl;

	std::vector<float> edgeLengths;
	edgeLengths.reserve(20);  // prob. never going over 20 subdivisions;
	float currentEdgeLength = maxEdgeLength;
	// each octree level the edge length gets halfed, continue until we reach minEdgeLength
	// dr. seidel minEdgeLength 0.10
	// edge length at level n (starting at 0, otherwise n-1 when starting at 1) = maxEdgeLength * (1 / 2 ^ n);
	for (int i = 1; currentEdgeLength >= minEdgeLength; ++i)
	{
		edgeLengths.push_back(currentEdgeLength);
		currentEdgeLength = maxEdgeLength * 1 / (1 << i);
		//currentEdgeLength /= 2.0f;
	}

	// print edge lengths
	std::cout << "Edge lengths: " << std::endl;
	print_vec(edgeLengths);

	std::vector<size_t> voxelsRequired2;
	voxelsRequired2.reserve(20);
	// size of this vector after getOccupiedVoxelCenters equals amount of occupied voxels at specified octree resoultion(voxel side length)
	std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>> occupiedVoxelCenters;
	pcl::octree::OctreePointCloudOccupancy<pcl::PointXYZ> octOcc(edgeLengths[0]);
	for (float edgeLength : edgeLengths)
	{
		// octree needs to be empty to set resolution
		octOcc.setResolution(edgeLength);
		// add all points of cloud to check occupancy
		octOcc.setOccupiedVoxelsAtPointsFromCloud(cloud);

		// check occupancy
		octOcc.getOccupiedVoxelCenters(occupiedVoxelCenters);
		voxelsRequired2.push_back(occupiedVoxelCenters.size());

		// empty octree so we can set to different resolution
		octOcc.deleteTree();
	}
	std::cout << "Number of used voxels (using PCL OctreeOccupancy):" << std::endl;
	print_vec(voxelsRequired2);

}

// argv: 0: -, 1: minEdgeLength, 2: method (seidel, cc, pcl) 3: path to input file either ascii or pcd
int main(int argc, char* argv[])
{
	if (argc < 4)
	{

		std::cout << "Usage: boxdim minEdgeLength methodName inputFilename" << std::endl;
		std::cout << "       methods: seidel, cc, pcl" << std::endl;
		return 1;
	}
	float minEdgeLength = atof(argv[1]);
	if (minEdgeLength <= 0)
	{
		std::cout << "Minimal edge length needs to be > 0" << std::endl;
		return 1;
	}

	std::string input_filename(argv[3]);
	std::string base_name = input_filename.substr(0, input_filename.find_last_of('.'));
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (boost::ends_with(input_filename, ".pcd"))
	{
		std::cout << "Reading from pcd file: " << input_filename << std::endl;
		pcl::PCDReader reader;
		reader.read(input_filename, *cloud);
	}
	else
	{
		std::cout << "Reading from (assumed) ascii file: " << input_filename << std::endl;
		std::ifstream infile(argv[2]);
		float x, y, z;
		while (infile >> x >> y >> z)
		{
			pcl::PointXYZ point;
			point.x = x;
			point.y = y;
			point.z = z;
			cloud->insert(cloud->end(), point);
			// ignore rgb channel values, max 255 chars are ignored or till we find a \n
			infile.ignore(255, '\n');
		}
		infile.close();
	}

	if (strcmp(argv[2], "seidel") == 0)
	{
		std::cout << "==== USING DR. SEIDEL METHOD ====" << std::endl;
		box_dim_seidel(cloud, base_name, minEdgeLength);
	}
	else if (strcmp(argv[2], "cc") == 0)
	{
		std::cout << "==== USING CloudCompare METHOD ====" << std::endl;
		box_dim_cc(cloud, base_name, minEdgeLength);
	}
	else
	{
		std::cout << "==== USING PointCloudLibrary METHOD ====" << std::endl;
		box_dim_pcl(cloud, base_name, minEdgeLength);
	}
	//// find min/max coordinates manually
	//float min_x, min_y, min_z;
	//min_x = min_y = min_z = std::numeric_limits<float>::max();
	//float max_x, max_y, max_z;
	//max_x = max_y = max_z = std::numeric_limits<float>::lowest();
	//for (auto pt : *cloud)
	//{
	//	min_x = min_x > pt.x ? pt.x : min_x;
	//	min_y = min_y > pt.y ? pt.y : min_y;
	//	min_z = min_z > pt.z ? pt.z : min_z;
	//	max_x = max_x < pt.x ? pt.x : max_x;
	//	max_y = max_y < pt.y ? pt.y : max_y;
	//	max_z = max_z < pt.z ? pt.z : max_z;
	//}
	//std::cout << "Manual min: " << min_x << " " << min_y << " " << min_z << std::endl;
	//std::cout << "Manual max: " << max_x << " " << max_y << " " << max_z << std::endl;
	//std::cout << "Volume: " << (max_x - min_x) * (max_y - min_y) * (max_z - min_z) << std::endl;

	// PCL OCTREE TEST

	// constructor takes resolution: Octree resolution - side length of octree voxels
	//pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(edgeLengths[edgeLengths.size() - 1]);
	//octree.setInputCloud(cloud);
	//octree.addPointsFromInputCloud();

	// iterate over nodes
	//for (auto it = octree.begin(); it != octree.end(); ++it) {
	//	if (it.isBranchNode()) {
	//		std::cout << "branch" << std::endl;
	//	}

	//	if (it.isLeafNode()) {
	//		std::cout << "leaf" << std::endl;
	//	}
	//}

	return 0;
}
