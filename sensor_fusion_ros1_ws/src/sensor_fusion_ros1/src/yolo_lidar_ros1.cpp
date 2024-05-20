#include<iostream>
#include<fstream>
#include<ros/ros.h>
#include<std_msgs/String.h>
#include<sensor_msgs/PointCloud2.h>
#include<sensor_msgs/Image.h>
#include<vision_msgs/Detection2DArray.h>
#include<pcl/point_types.h>
#include<pcl/PCLPointCloud2.h>
#include<pcl_conversions/pcl_conversions.h>
#include<pcl/common/transforms.h>
#include<pcl/common/common.h>
#include<pcl_ros/filters/filter.h>
#include<pcl_ros/point_cloud.h>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>
#include <boost/thread/thread.hpp>
#include <numeric>
#include <vector>
#include <algorithm>
#include <functional>
#include <stdlib.h>
#include <boost/bind.hpp>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/search/kdtree.h>
#include <unordered_map>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/time.h>
#include<opencv2/calib3d.hpp> //Need to include PCL functions before opencv, otherwise flann errors may occur
#include<opencv2/core/mat.hpp>
#include<opencv2/core/eigen.hpp>
#include<opencv2/opencv.hpp>
#include<cv_bridge/cv_bridge.h>
#include<opencv2/highgui/highgui.hpp>
#include <chrono>
#include <ctime>

 
using namespace std;
using namespace cv;
ros::Publisher pub;
ros::Publisher pc_pub;

string getCurrentTimestamp() {
    // get current time
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);

    std::string timestamp = std::ctime(&time);
    timestamp.erase(timestamp.find_last_not_of("\n") + 1);

    for (char& c : timestamp) {
        if (c == ' ' || c == ':') {
            c = '_';
        }
    }
    return timestamp;
}

double truncatedMean(const std::vector<double>& nums, double threshold) {
    std::vector<double> sortedNums = nums;  // copy the input vector
    std::sort(sortedNums.begin(), sortedNums.end());  // sort the input vector
    // calculate the number of points to be truncated
    int trimCount = static_cast<int>(threshold * sortedNums.size());
    // truncate the middle part of the input vector
    //std::vector<double> trimmedNums(sortedNums.begin() + trimCount, sortedNums.end() - trimCount);
    std::vector<double> trimmedNums(sortedNums.begin(), sortedNums.end() - trimCount);
    // calculate the truncated mean value
    double mean = accumulate(trimmedNums.begin(), trimmedNums.end(), 0.0) / trimmedNums.size();
    return mean;
}

double getMinAbsValue(const std::vector<double>& nums) {
    // create a vector to store the abs value
    std::vector<double> absNums(nums.size());
    std::transform(nums.begin(), nums.end(), absNums.begin(), [](double num) {
        return std::abs(num);
    });
    // find the min value
    auto minIter = std::min_element(absNums.begin(), absNums.end());
    int minIndex = std::distance(absNums.begin(), minIter);
    // return the original value
    return nums[minIndex];
}
 
void PointCallback(const sensor_msgs::PointCloud2::ConstPtr& point_msg,
    const vision_msgs::Detection2DArray::ConstPtr& detection_msg,
    const float& ground_level,
    const float& leaf_size,
    const float& cluster_tolerance,
    const int& MinClusterSize,
    const float& truncate_threshold,
    const float& bbox_rescale,
    const string& filename) {
    // check if there is valid detections, if yes, then proceed, if no, then skip
    if (!detection_msg->detections.empty())
    {
        // 定义下采样后的点云和聚类结果 Define point cloud after downsampling and clustering results 
        // Define point type as pcl::PointXYZRGB for the point cloud message
        // Convert ROS message to PointCloud message
        pcl::console::TicToc tt, tt1;
        tt.tic();
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_msg(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*point_msg, *cloud_msg);
        tt1.tic();        
        
        // 过滤地面点（PassThrough滤波）
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr x_removed(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PassThrough<pcl::PointXYZRGB> passthrough_x;
        passthrough_x.setInputCloud(cloud_msg);
        passthrough_x.setFilterFieldName("x");
        passthrough_x.setFilterLimits(-4, 4);
        passthrough_x.filter(*x_removed);
        
        // 过滤地面点（PassThrough滤波）
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr ground_removed(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PassThrough<pcl::PointXYZRGB> passthrough;
        passthrough.setInputCloud(x_removed);
        passthrough.setFilterFieldName("z");
        passthrough.setFilterLimits(ground_level, 10.0);
        passthrough.filter(*ground_removed);
        
        // Define a new PointCloud variable for the downsampled cloud
        // Define a vector of PointIndices for the cluster indices
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr downsampled_cloud_msg(new pcl::PointCloud<pcl::PointXYZRGB>);
        
        // Define VoxelGrid objects, set the input cloud to the original cloud message
        // Set the voxel size for the downsampled cloud (0.1x0.1x0.1) meter
        // Apply the filter method to obtain the downsampled cloud message
        // The number of points in the point cloud can be reduced
        // Improving the processing speed of the point cloud and reducing the demand for computing resources
        // We can also use the orginal point cloud directly, more tests are needed to see the impacts
        std::cerr << "PointCloud before filtering: " << cloud_msg->width * cloud_msg->height
            << " data points." << std::endl;
            //(" << pcl::getFieldsList (*cloud_msg) << ")
        
        pcl::VoxelGrid<pcl::PointXYZRGB> sor;
        sor.setInputCloud(ground_removed);
        sor.setLeafSize(leaf_size, leaf_size, leaf_size);
        sor.filter(*downsampled_cloud_msg);
        
        std::cerr << "PointCloud after filtering: " << downsampled_cloud_msg->width * downsampled_cloud_msg->height
            << " data points." << std::endl;
            //(" << pcl::getFieldsList (*downsampled_cloud_msg) << ")
        
        // Define a new PointCloud message for the filtered cloud
        // Copy the downsampled cloud message to the new cloud message
        //pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        //pcl::copyPointCloud(*downsampled_cloud_msg, *cloud);
        
        // Define a KdTree object and set the input cloud
        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
        tree->setInputCloud(downsampled_cloud_msg);
        //Create a vector of PointIndices, which contain the actual index information in a vector<int>. 
        //The indices of each detected cluster are saved here
        std::vector<pcl::PointIndices> cluster_indices;
        
        // Define an EuclideanClusterExtraction object and set the parameters
        pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> ec;
        ec.setClusterTolerance(cluster_tolerance); //设置聚类距离阈值 Set clustering distance threshold
        ec.setMinClusterSize(MinClusterSize); //设置最小聚类点数 Set the minimum number of cluster points
        ec.setMaxClusterSize(downsampled_cloud_msg->size()/2); //设置最大聚类点数 Set the maximum number of cluster points
        ec.setSearchMethod(tree);// Use KDTree as the search method
        ec.setInputCloud(downsampled_cloud_msg);// Apply clustering on the downsampled cloud
        ec.extract(cluster_indices);// Extract the cluster indices from the cloud
        float Preprocessing_time = tt1.toc();
        std::cerr << "Preprocess done,cost " << Preprocessing_time << " ms," << std::endl;
        std::cerr << "Found " << cluster_indices.size() << " clusters." << std::endl;
        
        // initialize and define points_3d for point cloud vector, points_2d for projected coordinates, distances for range information
        vector<Point3f> points_3d;
        vector<Point2f> points_2d;
        vector<double> x_list;
        vector<double> y_list;
        vector<double> distances;
        double range; //range is the distance for each point
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud_msg(new pcl::PointCloud<pcl::PointXYZRGB>);
        int n = 0;
        // 遍历每个聚类 Iterate over each cluster
        for (const auto &indices : cluster_indices)
        {
            float r = 255. * std::rand() / RAND_MAX;
            float g = 255. * std::rand() / RAND_MAX;
            float b = 255. * std::rand() / RAND_MAX;
            //std::cerr << "The No." << n << " cluster has " << indices.indices.size() << " points." << std::endl;
            //n++;
            // 遍历聚类中的每个点 Iterate over each point in the cluster     
            for (const auto index : indices.indices)
            {
            	downsampled_cloud_msg->at(index).r = r;
            	downsampled_cloud_msg->at(index).g = g;
            	downsampled_cloud_msg->at(index).b = b;
            	clustered_cloud_msg->push_back((*downsampled_cloud_msg)[index]);
        		pcl::PointXYZRGB pt = downsampled_cloud_msg->points[index]; //retrieve the 3D coordinates of the current point in the point cloud using its index. 
        		range = (double)sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z); //calculate the distance of each point 
        		distances.push_back(range); //and build distances vector for future reference
        		x_list.push_back(-pt.x);
        		y_list.push_back(-pt.y);
        		// build points_3d vector, transform point cloud to Point3f vector for projection
        		points_3d.push_back(cv::Point3f(pt.x, pt.y, pt.z));
            }
        }
        clustered_cloud_msg->width = clustered_cloud_msg->size();
        clustered_cloud_msg->height = 1;
        clustered_cloud_msg->is_dense = true;
        std::cerr << "PointCloud after clustering: " << clustered_cloud_msg->size() << " data points" << std::endl;
        //ROS_INFO("size of points_3d = %ld", points_3d.size());
        sensor_msgs::PointCloud2 msg_cloud_filter_cluster;
        //Publish the clustered cloud msg or only the downsampled cloud msg for visualization
        pcl::toROSMsg(*clustered_cloud_msg, msg_cloud_filter_cluster);
        //pcl::toROSMsg(*downsampled_cloud_msg, msg_cloud_filter_cluster);
        msg_cloud_filter_cluster.header.frame_id = "PandarXT-32";
        pc_pub.publish(msg_cloud_filter_cluster);        
        
        //create a copy of source_img in detection_msg for point cloud visualization
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(detection_msg->detections[0].source_img);
        // initialize and define points_2d for projected coordinates, distances for range information
        // define the parameters for calcuating the distance
        // Get the projection matrix and intrinsic matrix
        Mat intrinsic = (cv::Mat_<double>(3, 3) << 521.766, 0, 638.569,
                                                 0, 521.766, 356.64,
                                                 0, 0, 1);  // intrinsic matrix of camera
        Mat distCoeffs = (cv::Mat_<double>(1, 5) << 0, 0, 0, 0, 0); // distortion coefficients of camera
        Mat rotation = (cv::Mat_<double>(1, 3) << 0.0061, 2.2445, -2.1959);   // rotation matrix of camera
        Mat translation = (cv::Mat_<double>(1, 3) << 0.0654, -0.0781, -0.0458); // translation matrix of camera
        //use CV::projectPoints to apply the 3D to 2D projection, points_3d is the input, points_2d is the output
        projectPoints(points_3d, rotation, translation, intrinsic, distCoeffs, points_2d);
        
        // initialize the variables for the bbox
        ros::Time stamp = detection_msg->header.stamp;
        stringstream ss;
        ss << stamp.sec << "." << stamp.nsec;
        string timestamp = ss.str();
        int seq_id;
        int class_id;
        double score;
        float center_x;
        float center_y;
        float size_x;
        float size_y;
        int detection_Num = detection_msg->detections.size();
        std::cerr << "Found " << detection_Num << " detections." << std::endl;
        string csvLine;
        bool projected = false;
        int current_detection = 1;
        ofstream file(filename, std::ios::app);
        // iterate in the detections to get the distance for each detection
        for(const auto& detection : detection_msg->detections)
        {
            vector<double> distance;
            vector<double> x_output_list;
            vector<double> y_output_list;
            //initialize the min and avg distance as 0
            double minDistance = 0;
            double avgDistance = 0;
            double min_X = 0;
            double min_Y = 0;
            double avgX = 0;
            double avgY = 0;            
            //iterate in the 2d projected pixels
            //ROS_INFO("size of points_2d = %ld", points_2d.size());
            for (int i = 0; i < points_2d.size(); ++i)
            {
                //check if the pixel is inside the bbox
                //comment this "if" to visualize every point
                //if (points_2d[i].x >= detection.bbox.center.x-detection.bbox.size_x/2 && points_2d[i].x <= detection.bbox.center.x+detection.bbox.size_x/2 &&
                    //points_2d[i].y >= detection.bbox.center.y-detection.bbox.size_y/2 && points_2d[i].y <= detection.bbox.center.y+detection.bbox.size_y/2)
                if (points_2d[i].x >= detection.bbox.center.x-bbox_rescale*detection.bbox.size_x/2 && points_2d[i].x <= detection.bbox.center.x+bbox_rescale*detection.bbox.size_x/2 &&
                    points_2d[i].y >= detection.bbox.center.y-bbox_rescale*detection.bbox.size_y/2 && points_2d[i].y <= detection.bbox.center.y+bbox_rescale*detection.bbox.size_y/2)
                {
                    distance.push_back(distances[i]);//save the distance info if the pixel is inside
                    cv::circle(cv_ptr->image, points_2d[i], 1, cv::Scalar(255,0,0), -1);//paint the pixel in the image for visualization
                    x_output_list.push_back(x_list[i]);
                    y_output_list.push_back(y_list[i]);
                }
            }
            //the detection info
            seq_id = detection.header.seq;
            class_id = detection.results.back().id;
            score = detection.results.back().score;
            center_x = detection.bbox.center.x;
            center_y = detection.bbox.center.y;
            size_x = detection.bbox.size_x;
            size_y = detection.bbox.size_y;
            // get the min distance of the bbox
            if (!distance.empty()) {
        		projected = true;
        		minDistance = *min_element(distance.begin(), distance.end());
        		avgDistance = truncatedMean(distance, truncate_threshold);
        		min_X = getMinAbsValue(x_output_list);
        		min_Y = getMinAbsValue(y_output_list);
        		avgX = accumulate(x_output_list.begin(), x_output_list.end(), 0.0) / x_output_list.size();
        		avgY = truncatedMean(y_output_list, truncate_threshold);
        		// output the detection info with distance info
        		ROS_INFO("Detection info: seq_id:[%d], class_id:[%d], score:[%f], Min_distance:[%f], Avg_distance:[%f]", seq_id, class_id, score, minDistance, avgDistance);
        		ROS_INFO("Coordinate info: Min_X:[%f], Min_Y:[%f], Average_X:[%f], Average_Y:[%f]", min_X, min_Y, avgX, avgY);
        		if (current_detection < detection_Num) {
        		    csvLine = to_string(seq_id) + "," + timestamp + "," + to_string(class_id) + "," + to_string(score) + "," + to_string(minDistance) + "," + to_string(avgDistance) + "," + to_string(min_X) + "," + to_string(avgX) + "," + to_string(min_Y) + "," + to_string(avgY);
        		    //ofstream file(filename, std::ios::app);
        		    file << csvLine << endl;
		    //file.close();
		} else {
		    csvLine = to_string(seq_id) + "," + timestamp + "," + to_string(class_id) + "," + to_string(score) + "," + to_string(minDistance) + "," + to_string(avgDistance) + "," + to_string(min_X) + "," + to_string(avgX) + "," + to_string(min_Y) + "," + to_string(avgY) + "," + to_string(cloud_msg->width * cloud_msg->height) + "," + to_string(downsampled_cloud_msg->width * downsampled_cloud_msg->height) + "," + to_string(clustered_cloud_msg->size()) + "," + to_string(Preprocessing_time);
		}
		    //ROS_INFO("Bounding Box info: center_point:[%f,%f], size_x:[%f], size_y:[%f]", center_x, center_y, size_x, size_y);
            } else {
            	if (current_detection < detection_Num) {
		    csvLine = to_string(seq_id) + "," + timestamp + "," + to_string(class_id) + "," + to_string(score) + "," + "N/A" + "," + "N/A" + "," + "N/A" + "," + "N/A" + "," + "N/A" + "," + "N/A";
		    //ofstream file(filename, std::ios::app);
		    file << csvLine << endl;
		    //file.close();
		    } else {
		    csvLine = to_string(seq_id) + "," + timestamp + "," + to_string(class_id) + "," + to_string(score) + "," + "N/A" + "," + "N/A" + "," + "N/A" + "," + "N/A" + "," + "N/A" + "," + "N/A" + "," + to_string(cloud_msg->width * cloud_msg->height) + "," + to_string(downsampled_cloud_msg->width * downsampled_cloud_msg->height) + "," + to_string(clustered_cloud_msg->size()) + "," + to_string(Preprocessing_time);
		    }
            }
            current_detection++;
        }
        float Fusion_time = tt.toc();
        std::cerr << "Fusion done,cost " << Fusion_time << " ms," << std::endl;
        if (projected) {
	    //ofstream file(filename, std::ios::app);
	    file << csvLine << "," << to_string(Fusion_time) << endl;
	    file.close();
        } else {
	    file.close();
        }
        //publish the image for visualization
        pub.publish(cv_ptr->toImageMsg());
    }
}
 
int main (int argc, char **argv)
{
	ros::init (argc, argv, "yolo_lidar_ros1");
	ros::NodeHandle nh("~");
	float ground_level, leaf_size, cluster_tolerance, X_cut;
	double truncate_threshold, bbox_rescale;
	int MinClusterSize;
	nh.param<float>("ground_level", ground_level, -2.0);
	nh.param<float>("leaf_size", leaf_size, 0.1);
	nh.param<float>("cluster_tolerance", cluster_tolerance, 0.35);
	nh.param<double>("truncate_threshold", truncate_threshold, 0.2);
	nh.param<double>("bbox_rescale", bbox_rescale, 0.9);
	nh.param<int>("MinClusterSize", MinClusterSize, 20);
	ROS_INFO("Parameters: ground_level:[%f], leaf_size:[%f], cluster_tolerance:[%f], MinClusterSize:[%d], truncate_threshold:[%f], bbox_rescale:[%f]", ground_level, leaf_size, cluster_tolerance, MinClusterSize, truncate_threshold, bbox_rescale);
	pub = nh.advertise<sensor_msgs::Image>("/projectedImg", 100, true);
	pc_pub = nh.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("/pcl_points", 100, true);
	message_filters::Subscriber<sensor_msgs::PointCloud2> subscriber_pcl(nh,"/hesai/pandar",1,ros::TransportHints().tcpNoDelay());
    	message_filters::Subscriber<vision_msgs::Detection2DArray> subscriber_bbox(nh,"/yolov7",1,ros::TransportHints().tcpNoDelay());
	ROS_INFO("Inintiating");
	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, vision_msgs::Detection2DArray> syncPolicy;
	message_filters::Synchronizer<syncPolicy> sync(syncPolicy(10), subscriber_pcl, subscriber_bbox);//10 is the queue size
	ROS_INFO("Synchronizer defined");
	string filename = "/home/nvidia/Data/data_" + getCurrentTimestamp() + ".csv";
	ofstream file(filename, std::ios::app);
	if (!file.is_open()) {
		std::cerr << "Failed to open CSV file!";
		exit(EXIT_FAILURE);
	}
	file << "seq_id,timestamp,class_id,score,Min_distance,Avg_distance,Min_X,Average_X,Min_Y,Average_Y,Original_points,Downsampled_points,Clustered_points,Preprocessing_time,Fusion_time" << endl;
	file.close();
	ROS_INFO("CSV file created");
	sync.registerCallback(boost::bind(&PointCallback, _1, _2, ground_level, leaf_size, cluster_tolerance, MinClusterSize, truncate_threshold, bbox_rescale, filename));
	ROS_INFO("Callback registered");
	ros::spin();
	return 0;
}
