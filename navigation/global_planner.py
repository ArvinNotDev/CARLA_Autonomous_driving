import carla

import sys
sys.path.append(r"I:\CARLA_0.9.16\PythonAPI\carla") # locate the carla folder
sys.path.append(r"I:\CARLA_0.9.16\PythonAPI") # locate the carla folder

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.local_planner import RoadOption


class RoutePlanner:
    def __init__(self, world, resolution=2.0):
        self.world = world
        self.map = world.get_map()
        self.grp = GlobalRoutePlanner(
            self.map,
            resolution
        )

    def get_route(self, start_location, end_location):
        return self.grp.trace_route(
            start_location,
            end_location
        )

    def get_next_maneuver(self, current_location, goal_location):

        route = self.get_route(
            current_location,
            goal_location
        )

        for wp, option in route:
            if option != RoadOption.LANEFOLLOW:
                return option

        return None

    def get_next_maneuver_text(self, current_location, goal_location):

        option = self.get_next_maneuver(
            current_location,
            goal_location
        )

        if option == RoadOption.LEFT:
            return "LEFT"

        if option == RoadOption.RIGHT:
            return "RIGHT"

        if option == RoadOption.STRAIGHT:
            return "STRAIGHT"

        if option == RoadOption.CHANGELANELEFT:
            return "CHANGE_LANE_LEFT"

        if option == RoadOption.CHANGELANERIGHT:
            return "CHANGE_LANE_RIGHT"

        return "NONE"

    def distance_to_next_maneuver(self, current_location, goal_location):

        route = self.get_route(
            current_location,
            goal_location
        )

        distance = 0.0

        for i in range(len(route) - 1):

            wp1 = route[i][0]
            wp2 = route[i + 1][0]

            distance += wp1.transform.location.distance(
                wp2.transform.location
            )

            option = route[i][1]

            if option != RoadOption.LANEFOLLOW:
                return distance

        return None

    def get_next_instruction(self, current_location, goal_location):

        return {
            "maneuver": self.get_next_maneuver_text(
                current_location,
                goal_location
            ),
            "distance": self.distance_to_next_maneuver(
                current_location,
                goal_location
            )
        }