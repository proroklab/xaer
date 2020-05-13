#ifdef _WIN32
#include <cstdlib>
#endif

#include "ffpython.h"

int main(int argc, char* argv[])
{
	typedef vector<float> tuple_t;
	typedef vector<tuple_t> list_tuple_t;
	
	ffpython_t ffpython;
	PySys_SetArgv(argc, argv);
	ffpython.add_path("./py");
	printf("Loading module\n");
	ffpython.load("main");
	string map_file = "./map/aadc2018#test#track#003.xodr"; // OpenDrive format 1.4
	string maneuvers_file = "./maneuvers/example.xml";
	string sign_file = "./map/signs_aadc_testevent2018.xml";
	ffpython.call<void>("main", "init", map_file, maneuvers_file, sign_file);
	printf("Module loaded\n");
    
	// Build test input
	printf("Building input\n");
	tuple_t car_point = {226.89f,58.66f}; // changes every step
	float car_rotation = 1.57f; // changes every step
	tuple_t starting_car_point = car_point; // changes every time the maneuvers list changes
	string car_direction = "forward"; // forward: up is north; backward: up is south
	printf("Input built\n");
	
	// Generate path
	printf("Building path\n"); // build every time maneuvers list changes
    list_tuple_t path_roads = ffpython.call<list_tuple_t>("main", "get_path", starting_car_point, car_direction);
	printf("Generated path:\n");
	for (list_tuple_t::iterator it = path_roads.begin(); it != path_roads.end(); ++it)
	{
		tuple_t info = *it;
		printf("\tU: <%f,%f,%f,%f>, V: <%f,%f,%f,%f>, origin: <%f,%f>, rotation: <%f>, length: <%f>, inversion: <%f>\n", info[0], info[1], info[2], info[3], info[4], info[5], info[6], info[7], info[8], info[9], info[10], info[11], info[12]);
	}
	
	// Generate path from actions
	vector<string> actions = {"merge_left","right","left","park"};
	vector<int> extra_actions = {-1,-1,-1,2};
	for (int i = 0; i < actions.size(); ++i)
	{
		string action = actions[i];
		int extra_action = extra_actions[i];
		printf("Building path for action: %s\n", action.c_str());
		list_tuple_t action_roads = ffpython.call<list_tuple_t>("main", "get_path_by_action", action, extra_action, starting_car_point, car_direction);
		printf("Action '%s' path:\n", action.c_str());
		for (list_tuple_t::iterator at = action_roads.begin(); at != action_roads.end(); ++at)
		{
			tuple_t info = *at;
			printf("\tU: <%f,%f,%f,%f>, V: <%f,%f,%f,%f>, origin: <%f,%f>, rotation: <%f>, length: <%f>, inversion: <%f>\n", info[0], info[1], info[2], info[3], info[4], info[5], info[6], info[7], info[8], info[9], info[10], info[11], info[12]);
		}
	}
	
	// Setup splines for control point generation
	printf("Setup splines for control point generation.\n"); // setup every time path_roads changes
	ffpython.call<void>("main", "setup_splines_for_control_point_generation", path_roads);
	float old_car_progress = 0; // set to 0 after setup because car is automatically at the start of the path; set -1 if unknown; must be always lower than the real car progress in order to generate good results
	
	// Get closest road point to car, knowing old car progress may speed up algorithm (a lot)
	tuple_t closest_spline_point = ffpython.call<tuple_t>("main", "get_closest_road_point", car_point, old_car_progress);
	printf("Closest road point to car <%f,%f>\n", closest_spline_point[0],closest_spline_point[1]);
	// Get car progress on path, knowing old car progress may speed up algorithm (a lot)
	float car_progress = ffpython.call<float>("main", "get_path_progress", car_point, old_car_progress);
	printf("Car position on road %f\n", car_progress);
	// Get normalized car progress (in [0,1]) on path, knowing old car progress may speed up algorithm (a lot)
	float norm_car_progress = ffpython.call<float>("main", "get_normalized_path_progress", car_point, old_car_progress);
	printf("Maneuvers completion percentage: %f\n", norm_car_progress*100);
	// Generate control points
	float car_speed = 0.1;
	list_tuple_t control_points = ffpython.call<list_tuple_t>("main", "get_control_points", car_point, car_rotation, car_progress, car_speed);
	printf("Generated control points:\n");
	for (list_tuple_t::iterator it = control_points.begin(); it != control_points.end(); ++it)
	{
		tuple_t info = *it;
		printf("\tpoint: <%f,%f> %f %f\n", info[0], info[1], info[2], info[3]);
	}
#ifdef _WIN32
	system("pause");
#endif
    printf("main exit...\n");
    return 0;
}
