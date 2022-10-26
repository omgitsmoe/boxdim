#include <float.h>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

int main(int argc,char** argv)
{
	pcl::PCDReader reader;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	reader.read(argv[1],*cloud);

	float x_min, x_max, y_min, y_max;
	x_min = y_min = 0;
	x_max = y_max = FLT_MAX;

	float x, y;
	for(int i=0;i<cloud->points.size();i++)
	{
		x = cloud->points[i].x;
		y = cloud->points[i].y;
		// z = cloud->points[i].z;

		if (x < x_min)
			x_min = x;
		else if (x > x_max)
			x_max = x;
		if (y < y_min)
			y_min = y;
		else if (y > y_max)
			y_max = y;
	}
	std::cout << x_min << " " << x_max << " " << y_min << " " << y_max << std::endl;
	return 0;
}
