#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <limits>

#include <Eigen/Core>

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

// argv: 0: -, 1: voxel edge length for voxel grid downsampling (<0 disables downsampling),
//		 2: pcd file name, 3: smoothness
int main(int argc, char* argv[])
{
	if (argc < 3)
		return 1;
	float minEdgeLength = atof(argv[1]);
	if (minEdgeLength <= 0)
	{
		std::cout << "Minimal edge length needs to be > 0" << std::endl;
		return 1;
	}

	std::string input_filename(argv[2]);
	std::string base_name = input_filename.substr(0, input_filename.find_last_of('.'));
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (boost::ends_with(input_filename, ".pcd"))
	{
		std::cout << "Reading from pcd file: " << input_filename << std::endl;
		pcl::PCDReader reader;
		reader.read(argv[2], *cloud);
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
		*absDat = roundf(*absDat * 100.0f) / 100.0f;
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
	while (currentEdgeLength >= minEdgeLength)
	{
		edgeLengths.push_back(currentEdgeLength);
		currentEdgeLength /= 2.0f;
	}

	// print edge lengths
	std::cout << "Edge lengths: ";
	for (int i = 0; i < edgeLengths.size(); ++i) std::cout << edgeLengths[i] << ", ";
	std::cout << std::endl;

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

	if (edgeLengths.size() > relEdgeLengths_len)
	{
		std::cout << "ERROR: Not enough relative edge lengths (" << relEdgeLengths_len << " vs. " << edgeLengths.size() << ") available!!!" << std::endl;
		return 1;
	}
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
	std::cout << "Voxel idx factors: ";
	for (int i = 0; i < voxelIdxFactors.size(); ++i) std::cout << voxelIdxFactors[i] << ", ";
	std::cout << std::endl;

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
			currentVoxelIdx.x = static_cast<int>(/*floor*/(pt.x - min(0)) * voxelIdxFactors[i]);
			currentVoxelIdx.y = static_cast<int>(/*floor*/(pt.y - min(1)) * voxelIdxFactors[i]);
			currentVoxelIdx.z = static_cast<int>(/*floor*/(pt.z - min(2)) * voxelIdxFactors[i]);
			//currentVoxelIdx.x = roundf(pt.x * voxelIdxFactors[i] + 0.5f);
			//currentVoxelIdx.y = roundf(pt.y * voxelIdxFactors[i] + 0.5f);
			//currentVoxelIdx.z = roundf(pt.z * voxelIdxFactors[i] + 0.5f);
			usedVoxels.insert(currentVoxelIdx);
		}

		voxelsRequired.push_back(usedVoxels.size());
		usedVoxels.clear();
	}
	std::cout << "Number of used voxels: ";
	for (int i = 0; i < voxelsRequired.size(); ++i) std::cout << voxelsRequired[i] << ", ";
	std::cout << std::endl;

	std::string out_filename = base_name + ".csv";
	std::cout << "Writing output csv: " << out_filename << std::endl;

	// standard RFC 4180 uses comma as separator
	std::ofstream outfile(out_filename);
	for (int i = 0; i < voxelsRequired.size(); ++i)
	{
		// use left-shift instead of 2^i
		outfile << log(1 << i) << ", " << log((double)voxelsRequired[i]) << std::endl;
	}
	outfile.close();

	return 0;
}
