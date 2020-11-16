#include <string>
#include <set>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>

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

void writeCloudsSeparated(std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds, std::string fname)
{
	std::string base_name = fname.substr(0, fname.find_last_of('.')) + "_cl_";
	std::stringstream padded_nr;
	pcl::PCDWriter writer;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr out(new pcl::PointCloud<pcl::PointXYZRGB>);
	std::set<int> colours;
	std::cout << "Writing clouds to separate files!" << std::endl;

	for(int i=0;i<clouds.size();i++)
	{
		padded_nr.str("");
		// pad with 0 to width of 5
		padded_nr << std::setfill('0') << std::setw(5) << i;
		// make sure we use unique colours
		int r, g, b;
		int rgb;
		do {
			r = rand()%256;
			g = rand()%256;
			b = rand()%256;
			// 16 8 0 bit
			// RRGGBB
			rgb = (r << 16) | (g << 8) | (b << 0);
		} while (colours.count(rgb) > 0);

		for(int j=0; j<clouds[i]->points.size(); j++)
		{
			pcl::PointXYZRGB point;
			point.x = clouds[i]->points[j].x;
			point.y = clouds[i]->points[j].y;
			point.z = clouds[i]->points[j].z;
			point.r = r;
			point.g = g;
			point.b = b;
			out->insert(out->end(), point);
		}
		writer.write(base_name + padded_nr.str() + ".pcd", *out, true);
		out->clear();
	}
}

// merges clouds into one colouring all points from an origin cloud in the same colour, clears origin clouds by default
int mergeAndColorizeClouds(const std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& out_cloud, bool delete_origin = true)
{
	size_t totalPoints = 0;
	for (auto cloud : clouds) totalPoints += cloud->size();
	try
	{
		(*out_cloud).reserve(totalPoints);
	}
	catch (const std::bad_alloc&)
	{
		out_cloud->clear();
		return -1;
	}

	std::set<int> colours;
	for (int i = 0; i < clouds.size(); i++)
	{
		int r, g, b;
		int rgb;
		do {
			r = rand()%256;
			g = rand()%256;
			b = rand()%256;
			// 16 8 0 bit
			// RRGGBB
			rgb = (r << 16) | (g << 8) | (b << 0);
		} while (colours.count(rgb) > 0);

		for(size_t j=0; j<clouds[i]->points.size(); j++)
		{
			pcl::PointXYZRGB point;
			point.x = clouds[i]->points[j].x;
			point.y = clouds[i]->points[j].y;
			point.z = clouds[i]->points[j].z;
			point.r = r;
			point.g = g;
			point.b = b;
			out_cloud->insert(out_cloud->end(), point);
		}

		if (delete_origin)
			clouds[i]->clear();
	}

	return 0;
}

// find neares neighbour for all pts in origin and colour them the same colour as the found neares neighbour from other
void colourFromNearestPtInOtherCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &origin, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &other)
{
	pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtreeOther;
	kdtreeOther.setInputCloud(other);

	int K = 1;  // nr of neighbours to search for
	// need to be resized to K before passing to nearestKSearch
	std::vector<int> pointIdxNKNSearch(K);
	std::vector<float> pointNKNSquaredDistance(K);
	pcl::PointXYZRGB searchPoint;
	pcl::PointXYZRGB foundPoint;
	for(pcl::PointCloud<pcl::PointXYZRGB>::iterator it = origin->begin(); it != origin->end(); it++)
	{
		searchPoint.x = it->x;
		searchPoint.y = it->y;
		searchPoint.z = it->z;
		// nr of neighbours found returned
		if (kdtreeOther.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
		{
			foundPoint = other->at(pointIdxNKNSearch[0]);
			it->r = foundPoint.r;
			it->g = foundPoint.g;
			it->b = foundPoint.b;
		}
	}
}

// argv: 0: -, 1: voxel edge length for voxel grid downsampling (<0 disables downsampling),
//		 2: pcd file name, 3: smoothness
int main(int argc, char* argv[])
{
	std::string input_filename(argv[2]);
	float edgelength = atof(argv[1]);

	pcl::PCDReader reader;
	pcl::PCDWriter writer;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

	std::string base_name = input_filename.substr(0, input_filename.find_last_of('.'));
	std::string out_name = base_name + "_downsampled.pcd";

	// skip generating downsampled file if one already exists or deactivated with edgelength=0
	if (edgelength <= 0)
	{
		std::cout << "Reading input point cloud file... ";
		reader.read(argv[2], *cloud);
	}
	else
	{
		if (file_exists(out_name))
		{
			std::cout << "Reading downsampled input point cloud file... ";
			reader.read(out_name, *cloud);
		}
		else
		{
			// PCL uses boost smart pointers -> uses ref counting to see if memory can be freed
			pcl::PointCloud<pcl::PointXYZ>::Ptr original(new pcl::PointCloud<pcl::PointXYZ>);

			std::cout << "Reading input point cloud file... ";
			reader.read(argv[2], *original);
			std::cout << "Done!" << std::endl;

			std::cout << "Downsampling: " << base_name << std::endl;
			// method to reduce any localised variation
			// -> makes point cloud more uniform
			// benefit of downsampling in creating a more uniform dNN(z)
			// derived (euclidean) clusters are more uniform and less ‘patchy’
			// That is, they more distinctly partition the underlying surfaces whilst loss
			// of points in the upper canopy is reduced noticeably
			downsample(original, edgelength, cloud);
			original->clear();  // clear original cloud to save memory

			writer.write(out_name, *cloud, true);
			std::cout << "Wrote downsampled point cloud to " << out_name << std::endl;
		}
		
		base_name += "_downsampled";
	}
	std::cout << "Done!" << std::endl;

	std::cout << "Euclidean clustering: " << std::flush;
	// returns array of arrays of float where we have an array for every z-step (here 2)
	// those arrays consist of (z coord + z step, average mean dist to nnearest(here 50)
	// nearest neighbours)
	std::vector<std::vector<float>> nndata = dNNz(cloud, 50, 2);
	float nnmax = 0;
	// find max mean dist to nearest neighbours
	for (int i = 0; i < nndata.size(); i++) if (nndata[i][1] > nnmax) nnmax = nndata[i][1];
	std::cout << "max dNNz " << nnmax << ", " << std::flush;
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
	// euclideanClustering(cloud, dmax, nmin, clusters)
	// a set of point clouds are extracted whose constituent points have a common nearest
	// neighbour distance between at least one other point of less than or equal to **dmax**, and
	// number more than or equal to **Nmin**
	// dmax should be > dNN(z)
	// used to extract clusters representing topologically-connected surfaces
	// will be used as inputs for following feature extraction techniques
	euclideanClustering(cloud, nnmax, 3, clusters);

	out_name = base_name + "_euclidean-clusters.pcd";
	writeClouds(clusters, out_name, false);
	std::cout << out_name << std::endl;


	std::cout << "Region-based segmentation: " << std::flush;
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> regions;
	// finds index of cluster whose local minimal point is below a 2 unit range of the global
	// minimum point in a cloud of all clusters combined
	int idx = findPrincipalCloudIdx(clusters);
	// writer.write("PCA_CLUSTER.pcd", *clusters[idx], true);
	int nnearest = 50;
	int nmin = 3;
	float smoothness = atof(argv[3]);
	regionSegmentation(clusters[idx], nnearest, nmin, smoothness, regions);
	// BAD IDEA too many files! writeCloudsSeparated(regions, out_name);

	// already in original resolution
	if (edgelength <= 0)
	{
		out_name = base_name + "_ec_region-based-seg.pcd";
		writeClouds(regions, out_name, false);
		std::cout << out_name << std::endl;
		return 0;
	}

	std::cout << "Done!" << std::endl;

	// clear old stuff to save memory
	for (auto c : clusters) c->clear();
	clusters.clear();

	std::cout << "Merging regions into one cloud coloured by their region!" << std::endl;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr mergedRegions(new pcl::PointCloud<pcl::PointXYZRGB>);
	mergeAndColorizeClouds(regions, mergedRegions);  // already clears input clouds
	regions.clear();

	std::cout << "Loading original point cloud!" << std::endl;
	// load cloud in orginal resolution and infer regions from downsampled cloud
	// by finding a point's nearest neighbour
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr original(new pcl::PointCloud<pcl::PointXYZRGB>);
	reader.read(argv[2], *original);

	std::cout << "Infering point region colours by using a nearest neighbour search!" << std::endl;
	colourFromNearestPtInOtherCloud(original, mergedRegions);
	std::cout << "Writing output cloud in original resolutin: ";
	out_name = base_name + "_ec_region-based-seg_orig.pcd";
	writer.write(out_name, *original, true);
	std::cout << out_name << std::endl;

	return 0;
}
