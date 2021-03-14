#include <math.h>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <string>
#include <set>
#include <limits>
#include <chrono>  // for function timer

#include <immintrin.h>  // avx intrinsics
#include <xmmintrin.h>  // sse intrinsics

#include <Eigen/Core>

#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
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

std::vector<float> avgDistKNearest(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, const int K)
{
	std::vector<float> dist;
	std::vector<float> max_dist;
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);

	for(pcl::PointCloud<pcl::PointXYZ>::iterator it=cloud->begin();it!=cloud->end();it++)
	{
		pcl::PointXYZ searchPoint;
		searchPoint.x = it->x;
		searchPoint.y = it->y;
		searchPoint.z = it->z;

        // vectors for storing K neighbours' indices and squared distances
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<float> pointNKNSquaredDistance(K);

        std::vector<float> pointDists;
        pointDists.reserve(K);
        // search for K nearest neighbours
        if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 )
        {
            for (std::size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
                pointDists.push_back(sqrt(pointNKNSquaredDistance[i]));
        }
		float p_dist_sum = std::accumulate(pointDists.begin(), pointDists.end(), 0.0);
		float p_dist_mean = p_dist_sum / (float)pointDists.size();
		dist.push_back(p_dist_mean);

        // max_element returns an iterator -> needs to be dereferenced
        float p_dist_max = *std::max_element(pointDists.begin(), pointDists.end());
        max_dist.push_back(p_dist_max);
	}

	float dist_sum = std::accumulate(dist.begin(),dist.end(),0.0);
	float dist_mean = dist_sum/ (float)dist.size();
	float sq_sum = std::inner_product(dist.begin(), dist.end(), dist.begin(), 0.0);
	float stddev = std::sqrt(sq_sum / (float)dist.size() - dist_mean * dist_mean);

	float max_dist_sum = std::accumulate(max_dist.begin(),max_dist.end(),0.0);
	float max_dist_mean = max_dist_sum/ (float)max_dist.size();

    float overall_max_dist = *std::max_element(std::begin(max_dist), std::end(max_dist));

	std::vector<float> results;
	results.push_back(dist_mean);
	results.push_back(stddev);
    results.push_back(max_dist_mean);
    results.push_back(overall_max_dist);
	return results;
}

void slicePointCloud(
        pcl::PointCloud<pcl::PointXYZ>::Ptr &original,
        std::string dimension,
        float min, float max,
        pcl::PointCloud<pcl::PointXYZ>::Ptr &filtered)
{
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud(original);
    // axis x, y or z
	pass.setFilterFieldName(dimension);
	pass.setFilterLimits(min, max);

	pass.filter(*filtered);
}

// argv: 0: -, 1: minEdgeLength, 2: method (seidel, cc, pcl) 3: path to input file either ascii or pcd
int main(int argc, char* argv[])
{
	if (argc < 4)
	{

		std::cout << "Usage: pcstats <input_point_cloud_path> <K nearest neighbours> "
            << "<top % of tree crown>" << std::endl;
        std::cout << "    <K neares neighbours>: How many neighbours should be searched for "
            << "when calculating average/max nearest neighbour distance etc." << std:: endl;
        std::cout << "    <top % of tree crown>: Which part in percent from the top of the tree "
            << "should be used for calculating the same statistics as above" << std:: endl;
		return 1;
	}

	std::string input_filename(argv[1]);
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

	Eigen::Vector4f min, max;
	pcl::getMinMax3D(*cloud, min, max);

    // treeseg uses 50
    int knearest = atoi(argv[2]);
    std::cout << "Computing average and max distances using " << knearest
        << " as K-nearest neighbours!" << std::endl;
    constexpr float zstep = 2.0f;
    std::vector<std::vector<float>> results;
    // use tmp cloud so we don't have to re-allocate memory all the time when slicing
	pcl::PointCloud<pcl::PointXYZ>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZ>);
	for(float z=min[2]; z < max[2]; z += zstep)
	{
        // get z slice of original cloud
		slicePointCloud(cloud, "z", z, z+zstep, tmp);
		if(tmp->points.size() > knearest)
		{
            // average dist to knearest nearest neighbours, stddev
			std::vector<float> nn = avgDistKNearest(tmp, knearest);
			//float pos = z - min[2];
			float pos = z + zstep;
			std::vector<float> max_z_nndist;
			max_z_nndist.push_back(pos);
			max_z_nndist.push_back(nn[0]);
            max_z_nndist.push_back(nn[2]);
            max_z_nndist.push_back(nn[3]);
			results.push_back(max_z_nndist);
		}
		tmp->clear();
	}

    for(size_t i = 0; i < results.size(); ++i) {
        float max_z = results[i][0];
        float min_z = max_z - zstep;
        float avg_nndist = results[i][1];
        float avg_max_nndist = results[i][2];
        float max_nndist = results[i][3];

        std::cout << "Z-Slice(" << min_z << " -- " << max_z << "):" << std::endl;
        std::cout << "  Avg Dist: " << avg_nndist << std::endl;
        std::cout << "  Avg Max Dist: " << avg_max_nndist << std::endl;
        std::cout << "  Overall Max Dist: " << max_nndist << std::endl;
    }

    int crown_percent = atoi(argv[3]);
    float tree_length_factor = crown_percent / 100.0f;
    float height = max[2] - min[2];
    slicePointCloud(cloud, "z", max[2] - height * tree_length_factor, max[2], tmp);
    std::vector<float> nn_topcrown = avgDistKNearest(tmp, knearest);

    std::cout << "Top " << crown_percent << "% of the tree's crown:" << std::endl;
    std::cout << "  Avg Dist: "         << nn_topcrown[0] << std::endl;
    std::cout << "  Avg Max Dist: "     << nn_topcrown[2] << std::endl;
    std::cout << "  Overall Max Dist: " << nn_topcrown[3] << std::endl;
	return 0;
}
