//Andrew Burt - a.burt@ucl.ac.uk

#include <boost/algorithm/string/split.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

int main (int argc, char* argv[])
{
	pcl::PCDReader reader;
	pcl::PCDWriter writer;
	for(int i=1;i<argc;i++)
	{
		std::string tmp(argv[i]);
		std::vector<std::string> split;
		boost::split(split,tmp,boost::is_any_of("."));
		std::stringstream ss;
		for(int j=0;j<split.size()-1;j++) ss << split[j] << ".";
		ss << "txt";
		std::ofstream outfile(ss.str());
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		reader.read(argv[i],*cloud);
		for(int i=0;i<cloud->points.size();i++)
		{
			outfile << cloud->points[i].x << " " << cloud->points[i].y << " " << cloud->points[i].z << "\n";
		}
		outfile.close();
	}
	return 0;
}
