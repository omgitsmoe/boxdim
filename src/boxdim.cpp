#include <math.h>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <limits>
#include <chrono>  // for function timer

#if 0
#include <immintrin.h>  // avx intrinsics
#endif
#include <xmmintrin.h>  // sse intrinsics

#ifdef BOXDIM_WITH_OPEN_CL
// #define CL_TARGET_OPENCL_VERSION 110  // OpenCL c bindings: target other than most-current version (2.2)
// c bindings #include <CL/opencl.h>
// c++ compile-time settings for target/min version
#define CL_HPP_TARGET_OPENCL_VERSION 110
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#include <CL/cl2.hpp>
#endif

#include <Eigen/Core>

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#ifdef BOXDIM_WITH_PCL_OCT
#include <pcl/octree/octree_pointcloud.h>
#include <pcl/octree/octree_pointcloud_occupancy.h>
#endif

#ifdef BOXDIM_WITH_TREETOP
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#endif


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

void write_result(const std::vector<float>& edgeLengths, const std::vector<size_t>& voxelsRequired, const std::string& out_filename)
{
	// standard RFC 4180 uses comma as separator
	std::ofstream outfile(out_filename);
	for (int i = 0; i < voxelsRequired.size(); ++i)
	{
		// use left-shift instead of 2^i
		// outfile << log(1 << i) << ", " << log((double)voxelsRequired[i]) << std::endl;
		outfile << edgeLengths[i] << ", " << voxelsRequired[i] << std::endl;
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
	float relEdgeLengths[] = { 1.0f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f, 0.015625f, 0.0078125f, 0.00390625f,
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

	// TIMER START
	auto start = std::chrono::high_resolution_clock::now();
	//
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
	std::cout << "Number of used voxels:" << std::endl;
	print_vec(voxelsRequired);

	std::string out_filename = base_name + "_seidel.csv";
	std::cout << "Writing output csv: " << out_filename << std::endl;
	write_result(edgeLengths, voxelsRequired, out_filename);
}

#ifdef BOXDIM_WITH_OPEN_CL
void box_dim_seidel_gpu(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::string base_name, float minEdgeLength)
{
	//
	// OpenCL SETUP
	//

	// get one of the OpenCL platforms. This is actually a driver you had
	// previously installed. So platform can be from Nvidia, Intel, AMD....
	std::vector<cl::Platform> all_platforms;
	cl::Platform::get(&all_platforms);

	if (all_platforms.size() == 0) {
		std::cout << " No platforms found. Check OpenCL installation!\n";
		exit(1);
	}
	cl::Platform default_platform = all_platforms[0];
	std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << "\n";

	// get default device (CPUs, GPUs) of the default platform
	std::vector<cl::Device> all_devices;
	default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
	if (all_devices.size() == 0) {
		std::cout << " No devices found. Check OpenCL installation!\n";
		exit(1);
	}

	// use device[1] because that's a GPU; device[0] might be CPU?
	cl::Device default_device = all_devices[0];
	std::cout << "Using device: " << default_device.getInfo<CL_DEVICE_NAME>() << "\n";

	// a context is like a "runtime link" to the device and platform;
	// i.e. communication is possible
	cl::Context context({ default_device });

	//
	//
	//

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
	std::vector<float> voxelIdxFactors;
	voxelIdxFactors.reserve(20);  // prob. never going over 20 subdivisions;
	for (int i = 0; i < edgeLengths.size(); ++i)
	{
		voxelIdxFactors.push_back(1.0f / edgeLengths[i]);
	}
	std::cout << "Voxel idx factors:" << std::endl;
	print_vec(voxelIdxFactors);

	//
	// OpenCL compile program
	//

	// create the program that we want to execute on the device
	cl::Program::Sources sources;

	// only pointers can be qualified as __constant in kernel arguments
	// so int by value etc. is not allowed as const
	// Constant: A small portion of cached global memory visible by all workers.Use it if you can, read only.
	// Global : Slow, visible by all, read or write.It is where all your data will end, so some accesses to it are always necessary.
	// Local : Do you need to share something in a local group ? Use local!Do all your local workers access the same global memory ? Use local!Local memory is only visible inside local workers, and is limited in size, however is very fast.
	// Private : Memory that is only visible to a worker, consider it like registers.All non defined values are private by default.
	// What if i want to pass the length of an array? When you don't specify a qualifier in the kernel parameter it typically defaults to constant, which is what you want for those small elements, to have a fast access by all workers.
	std::string kernel_code =
		"   void kernel compute_voxel_idx(global const float* points,"
		"								  global float* vxIdxs,"
		"								  float vIdxFactor,"
		"                                 unsigned count) {"
		"       unsigned ID, Nthreads, ratio, start, stop;"
		""
		"       ID = get_global_id(0);"  // Returns the unique global work-item ID value for dimension identified by dimindx =^ thread id
		"       Nthreads = get_global_size(0);"
		""
		"       ratio = (count / Nthreads);"  // number of elements for each thread
		"       start = ratio * ID;"
		"       stop  = ratio * (ID + 1);"
		""
		"       for (unsigned i=start; i<stop; i++)"
		"           vxIdxs[i] = round(points[i] * vIdxFactor + 0.5f);"
		"   }";
	sources.push_back({ kernel_code.c_str(), kernel_code.length() });

	cl::Program program(context, sources);
	if (program.build({ default_device }) != CL_SUCCESS) {
		std::cout << "Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << std::endl;
		exit(1);
	}

	// create buffers on device (allocate space on GPU)
	// 16-bytes per pcl::PointXYZ due to sse-friendly alignment
	cl::Buffer buffer_Points(context, CL_MEM_READ_WRITE, sizeof(float) * 4 * cloud->size());
	cl::Buffer buffer_vxIdxs(context, CL_MEM_READ_WRITE, sizeof(float) * 4 * cloud->size());

	// create a queue (a queue of commands that the GPU will execute)
	cl::CommandQueue queue(context, default_device);
	// KernelFunctor gone use make_kernel for same functionality or use Kernel with setArg and queue.enqueueNDRange...
	cl::Kernel compute_voxel_idx(program, "compute_voxel_idx");
	//
	//
	//

	// TIMER START
	auto start = std::chrono::high_resolution_clock::now();
	//
	//
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

	// push write commands to queue
	// pcl::PointCloud uses vector for storing the Points and PointXYZ is just a struct with floats xyz and one float for padding to 16 bytes
	// so we can use a ptr to the start of cloud vector as float array ptr to fill buffer_Points
	queue.enqueueWriteBuffer(buffer_Points, CL_TRUE, 0, sizeof(float) * 4 * cloud->size(), &(*cloud)[0]);
	compute_voxel_idx.setArg(0, buffer_Points);
	compute_voxel_idx.setArg(1, buffer_vxIdxs);
	// !! IMPORTANT !! doesn't result in an error spent 2 hours looking for this bug
	// we have to cast size_t to unsigned int
	// 6.8 k. Arguments to kernel functions in a program cannot be declared with the built-in scalar types bool, half, size_t, ptrdiff_t, intptr_t, and uintptr_t
	compute_voxel_idx.setArg(3, static_cast<unsigned int>(cloud->size() * 4));  // float count
	std::vector<float> vxIdxsOut(cloud->size() * 4);

	for (int i = 0; i < edgeLengths.size(); ++i)
	{
		// set voxel index factor for this edge length
		compute_voxel_idx.setArg(2, voxelIdxFactors[i]);

		// kernel, offset, nr of threads
		queue.enqueueNDRangeKernel(compute_voxel_idx, cl::NullRange, cl::NDRange(300), cl::NullRange);
		queue.finish();

		// read result from GPU to vxIdxsOut
		queue.enqueueReadBuffer(buffer_vxIdxs, CL_TRUE, 0, sizeof(float) * 4 * cloud->size(), vxIdxsOut.data());
		// not needed since we passed CL_TRUE for blocking argument in enqueueReadBuffer // queue.finish();
		for (size_t j = 0; j < vxIdxsOut.size(); j += 4)
		{
			usedVoxels.insert({ vxIdxsOut[j], vxIdxsOut[j + 1], vxIdxsOut[j + 2] });  // last float just padding
		}

		voxelsRequired.push_back(usedVoxels.size());
		usedVoxels.clear();
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
	std::cout << "GPU TIMING: " << duration.count() << " milliseconds" << std::endl;
	//
	std::cout << "Number of used voxels:" << std::endl;
	print_vec(voxelsRequired);

	std::string out_filename = base_name + "_seidel.csv";
	std::cout << "Writing output csv: " << out_filename << std::endl;
	write_result(edgeLengths, voxelsRequired, out_filename);
}
#endif

#if 0
void box_dim_seidel_avx(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::string base_name, float minEdgeLength)
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
	std::vector<float> voxelIdxFactors;
	voxelIdxFactors.reserve(20);  // prob. never going over 20 subdivisions;
	for (int i = 0; i < edgeLengths.size(); ++i)
	{
		voxelIdxFactors.push_back(1.0f / edgeLengths[i]);
	}
	std::cout << "Voxel idx factors:" << std::endl;
	print_vec(voxelIdxFactors);

	// TIME START
	auto start = std::chrono::high_resolution_clock::now();
	//
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
	// needs to be 32-byte aligned since we use 256-bit avx registers
	alignas(32) float twoPtCoords[8];
	std::set<voxelIdx> usedVoxels;
	// set_ps is backwards!! (depending on how u look at it; since intel uses little-endian)
	__m256 halfScalar = _mm256_set_ps(0, 0, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f);
	for (int i = 0; i < edgeLengths.size(); ++i)
	{
		__m256 vIdxF = _mm256_set_ps(0, 0, voxelIdxFactors[i], voxelIdxFactors[i], voxelIdxFactors[i], voxelIdxFactors[i], voxelIdxFactors[i], voxelIdxFactors[i]);
		for (size_t pi = 0; pi < cloud->size(); pi += 2)
		{
			// dr. seidel method
			pcl::PointXYZ* pt1 = &(*cloud)[pi];
			pcl::PointXYZ* pt2 = &(*cloud)[pi + 1];
			__m256 ptxyz = _mm256_set_ps(0, 0, pt2->z, pt2->y, pt2->x, pt1->z, pt1->y, pt1->x);
			__m256 r = _mm256_mul_ps(ptxyz, vIdxF);
			r = _mm256_add_ps(r, halfScalar);
			// (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
			r = _mm256_round_ps(r, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
			// store values from register in float array since we use 256-bit registers the float array
			// needs to be 32-byte aligned
			// need to pass pointer to first float in float array otherwise compiler complains about
			// wrong type since we only need float* and not float[8]*
			_mm256_store_ps(&twoPtCoords[0], r);
			//std::cout << "VoxelIDX: " << twoPtCoords[0] << " " << twoPtCoords[1] << " " << twoPtCoords[2] << std::endl;
			usedVoxels.insert({ twoPtCoords[0], twoPtCoords[1], twoPtCoords[2] });
			usedVoxels.insert({ twoPtCoords[3], twoPtCoords[4], twoPtCoords[5] });
		}

		voxelsRequired.push_back(usedVoxels.size());
		usedVoxels.clear();
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
	std::cout << "AVX TIMING: " << duration.count() << " milliseconds" << std::endl;
	//
	std::cout << "Number of used voxels:" << std::endl;
	print_vec(voxelsRequired);

	std::string out_filename = base_name + "_seidel.csv";
	std::cout << "Writing output csv: " << out_filename << std::endl;
	write_result(edgeLengths, voxelsRequired, out_filename);
}
#endif

/*
 SSE version is fastest, since too much overhead in packing and unpacking into 32-byte aligned arrays etc.
 whereas the pcl::PointXYZ is already alligned for SSE

 NORMAL TIMING: 6797 milliseconds
 NORMAL TIMING: 6744 milliseconds
 NORMAL TIMING: 6645 milliseconds

 AVX TIMING: 3327 milliseconds
 AVX TIMING: 3353 milliseconds
 AVX TIMING: 3339 milliseconds

 SSE TIMING: 3090 milliseconds
 SSE TIMING: 3079 milliseconds
 SSE TIMING: 3081 milliseconds

 even GPU is slower with 700 threads!!
 GPU TIMING: 3644 milliseconds
 GPU TIMING: 3809 milliseconds 500 threads
 GPU TIMING: 4076 milliseconds 300 threads
*/
void box_dim_seidel_sse(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::string base_name, float minEdgeLength)
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
	std::vector<float> voxelIdxFactors;
	voxelIdxFactors.reserve(20);  // prob. never going over 20 subdivisions;
	for (int i = 0; i < edgeLengths.size(); ++i)
	{
		voxelIdxFactors.push_back(1.0f / edgeLengths[i]);
	}
	std::cout << "Voxel idx factors:" << std::endl;
	print_vec(voxelIdxFactors);

	// TIME START
	auto start = std::chrono::high_resolution_clock::now();
	//
	std::vector<size_t> voxelsRequired;
	voxelsRequired.reserve(20);

	// needed for sorting in set
	auto ptxyzLess = [](const pcl::PointXYZ& lhs, const pcl::PointXYZ& rhs)
	{
		// return (x < pt.x) || ((pt.x < x)) && (y < pt.y)) || ((!(pt.x < x)) && (!(pt.y < y)) && (z < pt.z));
		return (lhs.x < rhs.x) || ((lhs.x == rhs.x) && (lhs.y < rhs.y)) || ((lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z < rhs.z));
	};

	pcl::PointXYZ currentVoxelIdx;
	// type of comparator needed, so use decltype to get type of lambda
	// comparator function also needs to be passed to constructor!
	std::set<pcl::PointXYZ, decltype(ptxyzLess)> usedVoxels(ptxyzLess);
	// set_ps is backwards!! (depending on how u look at it; since intel uses little-endian)
	__m128 halfScalar = _mm_set_ps(0, 0.5f, 0.5f, 0.5f);
	for (int i = 0; i < edgeLengths.size(); ++i)
	{
		__m128 vIdxF = _mm_set_ps(0, voxelIdxFactors[i], voxelIdxFactors[i], voxelIdxFactors[i]);
		for (size_t pi = 0; pi < cloud->size(); ++pi)
		{
			// dr. seidel method
			// pcl::PointXYZ is already 16-byte aligned for SSE friendliness
			__m128 ptxyz = _mm_load_ps(reinterpret_cast<const float*>(&(*cloud)[pi]));
			__m128 r = _mm_mul_ps(ptxyz, vIdxF);
			r = _mm_add_ps(r, halfScalar);
			// (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
			r = _mm_round_ps(r, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
			// need to pass pointer to first float in float array otherwise compiler complains about
			// wrong type since we only need float* and not float[8]*
			_mm_store_ps(reinterpret_cast<float*>(&currentVoxelIdx), r);
			//std::cout << "VoxelIDX: " << twoPtCoords[0] << " " << twoPtCoords[1] << " " << twoPtCoords[2] << std::endl;
			usedVoxels.insert(currentVoxelIdx);
		}

		voxelsRequired.push_back(usedVoxels.size());
		usedVoxels.clear();
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
	std::cout << "SSE TIMING: " << duration.count() << " milliseconds" << std::endl;
	//
	std::cout << "Number of used voxels:" << std::endl;
	print_vec(voxelsRequired);

	std::string out_filename = base_name + "_seidel.csv";
	std::cout << "Writing output csv: " << out_filename << std::endl;
	write_result(edgeLengths, voxelsRequired, out_filename);
}

#ifdef BOXDIM_WITH_TREETOP
void box_dim_seidel_sse_treetop(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const std::string base_name, float minEdgeLength)
{
	Eigen::Vector4f min, max;
	// get min and max coordinates in each dimension which gives us the AABB
	pcl::getMinMax3D(*cloud, min, max);

	std::cout << "Min coordinates:\n" << min << "\nMax coordinates:\n" << max << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr zFilteredCloud(new pcl::PointCloud<pcl::PointXYZ>);
	// filter out lower 2/3 of the cloud in z/height
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z");
	float zmin = max.z() - 1.0f / 3.0f * (max.z() - min.z());
	std::cout << zmin << std::endl;
	pass.setFilterLimits(zmin, max.z());
	pass.filter(*zFilteredCloud);

	// swap target of shared_ptr (zFilteredCloud now points to the original cloud)
	cloud.swap(zFilteredCloud);
	zFilteredCloud.reset();  // swaps internal value it points to with nothing
	// re-compute min and max
	pcl::getMinMax3D(*cloud, min, max);

	std::cout << "New Min/Max after z filter:\nMin coordinates:\n" << min << "\nMax coordinates:\n" << max << std::endl;

	// get absolute dist between min and max coordinates to get the largest edge
	// length that we're gonna need for our octree
	Eigen::Vector4f absDist = (min - max).cwiseAbs();

	//std::cout << "Max edge length 3d:\n" << absDist << std::endl;

	// cloud already starts at 2/3 height since we filtered it before
	float mid_x = min.x() + (absDist.x() * 0.5f);
	float mid_y = min.y() + (absDist.y() * 0.5f);
	std::cout << "Mid x/y 2/3 z:" << mid_x << " " << mid_y << " " << min.z() << std::endl;
	float half_edge_length = 0.5f * absDist.z();

	// Eigen::Vector4f(maxX, maxY, maxZ, 1.0)
	min = { mid_x - half_edge_length, mid_y - half_edge_length, min.z(), 1.0f };
	max = { mid_x + half_edge_length, mid_y + half_edge_length, max.z(), 1.0f };

	std::cout << "Cropping cloud to\nMin:\n" << min << std::endl << "Max:\n" << max << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr croppedCloud(new pcl::PointCloud<pcl::PointXYZ>);
	// crop cloud to the top third in z and 0.5*height_third around middle x/y point 
	pcl::CropBox<pcl::PointXYZ> boxFilter;
	boxFilter.setMin(min);
	boxFilter.setMax(max);
	boxFilter.setInputCloud(cloud);
	boxFilter.filter(*croppedCloud);

	// use cropped cloud
	cloud.swap(croppedCloud);
	croppedCloud.reset();

	float maxEdgeLength = absDist.z();

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
	std::cout << "Edge lengths:" << std::endl;
	print_vec(edgeLengths);

	// inverse edge lengths -> multiplying a point's coordinates gives voxel indices for that point
	std::vector<float> voxelIdxFactors;
	voxelIdxFactors.reserve(20);  // prob. never going over 20 subdivisions;
	for (int i = 0; i < edgeLengths.size(); ++i)
	{
		voxelIdxFactors.push_back(1.0f / edgeLengths[i]);
	}
	std::cout << "Voxel idx factors:" << std::endl;
	print_vec(voxelIdxFactors);

	// TIME START
	auto start = std::chrono::high_resolution_clock::now();
	//
	std::vector<size_t> voxelsRequired;
	voxelsRequired.reserve(20);

	// needed for sorting in set
	auto ptxyzLess = [](const pcl::PointXYZ& lhs, const pcl::PointXYZ& rhs)
	{
		// return (x < pt.x) || ((pt.x < x)) && (y < pt.y)) || ((!(pt.x < x)) && (!(pt.y < y)) && (z < pt.z));
		return (lhs.x < rhs.x) || ((lhs.x == rhs.x) && (lhs.y < rhs.y)) || ((lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z < rhs.z));
	};

	pcl::PointXYZ currentVoxelIdx;
	// type of comparator needed, so use decltype to get type of lambda
	// comparator function also needs to be passed to constructor!
	std::set<pcl::PointXYZ, decltype(ptxyzLess)> usedVoxels(ptxyzLess);
	// set_ps is backwards!! (depending on how u look at it; since intel uses little-endian)
	__m128 halfScalar = _mm_set_ps(0, 0.5f, 0.5f, 0.5f);
	for (int i = 0; i < edgeLengths.size(); ++i)
	{
		__m128 vIdxF = _mm_set_ps(0, voxelIdxFactors[i], voxelIdxFactors[i], voxelIdxFactors[i]);
		for (size_t pi = 0; pi < cloud->size(); ++pi)
		{
			// dr. seidel method
			// pcl::PointXYZ is already 16-byte aligned for SSE friendliness
			__m128 ptxyz = _mm_load_ps(reinterpret_cast<const float*>(&(*cloud)[pi]));
			__m128 r = _mm_mul_ps(ptxyz, vIdxF);
			r = _mm_add_ps(r, halfScalar);
			// (_MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC) // round to nearest, and suppress exceptions
			r = _mm_round_ps(r, (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
			// need to pass pointer to first float in float array otherwise compiler complains about
			// wrong type since we only need float* and not float[8]*
			_mm_store_ps(reinterpret_cast<float*>(&currentVoxelIdx), r);
			//std::cout << "VoxelIDX: " << twoPtCoords[0] << " " << twoPtCoords[1] << " " << twoPtCoords[2] << std::endl;
			usedVoxels.insert(currentVoxelIdx);
		}

		voxelsRequired.push_back(usedVoxels.size());
		usedVoxels.clear();
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
	std::cout << "SSE TIMING: " << duration.count() << " milliseconds" << std::endl;
	//
	std::cout << "Number of used voxels:" << std::endl;
	print_vec(voxelsRequired);

	std::string out_filename = base_name + "_seidel_treetop.csv";
	std::cout << "Writing output csv: " << out_filename << std::endl;
	write_result(edgeLengths, voxelsRequired, out_filename);
}
#endif

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
	write_result(edgeLengths, voxelsRequired, out_filename);
}

#ifdef BOXDIM_WITH_PCL_OCT
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
#endif

// argv: 0: -, 1: minEdgeLength, 2: method (seidel, cc, pcl) 3: path to input file either ascii or pcd
int main(int argc, char* argv[])
{
    std::string min_edge_len;
    std::string method_name;
	std::string input_filename;
	if (argc == 1)
	{
        std::cout << "Minimum edge length (e.g. 0.1): ";
        std::cin >> min_edge_len;
        std::cout << "Method name (seidel, seidel_sse [recommended], cc): ";
        std::cin >> method_name;
        // cin cin >> may leave \n or whitespace etc. which will be interpreted by getline
        // as the user skipping the input
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
        std::cout << "Path to input file (ASCII *.txt or *.pcd), e.g. D:\\research\\b32.pcd:" << std::endl;
		// need to use getline since cin >> only reads till next whitespace
        std::getline(std::cin, input_filename);
    }
    else if (argc < 4)
	{

		std::cout << "Usage: boxdim minEdgeLength methodName inputFilename" << std::endl;
		// std::cout << "       methods: seidel (+ _avx, _sse, _sse_treetop, seidel_gpu), cc, pcl" << std::endl;
		std::cout << "       methods: seidel (+ _sse), cc" << std::endl;

		return 1;
	}
    else
    {
        min_edge_len = argv[1];
        method_name = argv[2];
        input_filename = argv[3];
    }

	float minEdgeLength = atof(min_edge_len.c_str());
	if (minEdgeLength <= 0)
	{
		std::cout << "Minimal edge length needs to be > 0" << std::endl;
		return 1;
	}

	auto start = std::chrono::high_resolution_clock::now();

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
		std::ifstream infile(argv[3]);
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

	if (method_name == "seidel")
	{
		std::cout << "==== USING DR. SEIDEL METHOD ====" << std::endl;
		box_dim_seidel(cloud, base_name, minEdgeLength);
	}
#if 0
	else if (strcmp(argv[2], "seidel_avx") == 0)
	{
		std::cout << "==== USING DR. SEIDEL METHOD WITH AVX INSTRUCTIONS ====" << std::endl;
		box_dim_seidel_avx(cloud, base_name, minEdgeLength);
	}
#endif
	else if (method_name == "seidel_sse")
	{
		std::cout << "==== USING DR. SEIDEL METHOD WITH SSE INSTRUCTIONS ====" << std::endl;
		box_dim_seidel_sse(cloud, base_name, minEdgeLength);
	}
#ifdef BOXDIM_WITH_TREETOP
	else if (method_name == "seidel_sse_treetop")
	{
		std::cout << "==== USING DR. SEIDEL METHOD WITH SSE INSTRUCTIONS ====" << std::endl;
		box_dim_seidel_sse_treetop(cloud, base_name, minEdgeLength);
	}
#endif
#ifdef BOXDIM_WITH_OPEN_CL
	else if (method_name == "seidel_gpu")
	{
		std::cout << "==== USING DR. SEIDEL METHOD ON THE GPU ====" << std::endl;
		box_dim_seidel_gpu(cloud, base_name, minEdgeLength);
	}
#endif
	else if (method_name == "cc")
	{
		std::cout << "==== USING CloudCompare METHOD ====" << std::endl;
		box_dim_cc(cloud, base_name, minEdgeLength);
	}
#ifdef BOXDIM_WITH_PCL_OCT
	else if (method_name == "pcl")
	{
		std::cout << "==== USING PointCloudLibrary METHOD ====" << std::endl;
		box_dim_pcl(cloud, base_name, minEdgeLength);
	}
#endif
	else
	{
		std::cout << "ERROR: Method not found, available methods are "
            << "seidel, seidel_sse, cc" << std::endl;
	}

	auto stop = std::chrono::high_resolution_clock::now();

	// Subtract stop and start timepoints and 
	// cast it to required unit. Predefined units 
	// are nanoseconds, microseconds, milliseconds, 
	// seconds, minutes, hours. Use duration_cast() 
	// function. 
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "TOTAL TIMING: " << duration.count() << " milliseconds" << std::endl;
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

    if (argc == 1) {
        // force to keep console window open for users that just clicked the exe
        std::cout << std::endl << "Press <ENTER> to close!" << std::endl;
        std::string dummy;
        std::getline(std::cin, dummy);
    }

	return 0;
}
