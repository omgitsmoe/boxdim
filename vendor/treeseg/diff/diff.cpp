#include <iostream>
#include <string>
#include <set>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

// needed for sorting in set
inline bool comp(const pcl::PointXYZ& lhs, const pcl::PointXYZ& rhs)
{
	// return (x < rhs.x) || ((rhs.x < x)) && (y < rhs.y)) || ((!(rhs.x < x)) && (!(rhs.y < y)) && (z < rhs.z));
	return (lhs.x < rhs.x) || ((lhs.x == rhs.x) && (lhs.y < rhs.y)) || ((lhs.x == rhs.x) && (lhs.y == rhs.y) && (lhs.z < rhs.z));
}

// writes points that are in new file but not in the old one to a pcd file
// argv: 0: -, 1: orig fn, 2: new fn
int main(int argc, char* argv[])
{
	std::string old_filename(argv[1]);
	std::string old_base_name = old_filename.substr(0, old_filename.find_last_of('.'));
	std::string new_filename(argv[2]);
	new_filename = new_filename.substr(new_filename.find_last_of('\\') + 1, new_filename.find_last_of('.'));

	pcl::PCDReader reader;
	pcl::PCDWriter writer;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr diff(new pcl::PointCloud<pcl::PointXYZ>);
	std::cout << "Reading input file for old point cloud: " << old_filename << std::endl;
	reader.read(argv[1], *cloud);

	// use comp as compare function, needs function pointer
	std::set<pcl::PointXYZ, decltype(comp)*> points(comp);
	for (auto pt : *cloud)
		points.insert(pt);
	cloud->clear();

	std::cout << "Reading input file for NEW point cloud: " << argv[2] << std::endl;
	reader.read(argv[2], *cloud);
	for (auto pt : *cloud)
	{
		if (points.count(pt) <= 0)
			diff->insert(diff->end(), pt);
	}

	std::string out_name = old_base_name + "_" + new_filename + "_diff.pcd";
	std::cout << "Writing " << out_name << std::endl;

	writer.write(out_name, *diff, true);
	std::cout << out_name << std::endl;

	return 0;
}
