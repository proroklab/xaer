from environments.utils.culture_lib.culture import Culture, Argument
from environments.car_controller.car_stuff.alex_discrete.road_cell import RoadCell
from environments.car_controller.car_stuff.alex_discrete.road_agent import RoadAgent
import numpy as np
import random

#####################
# EASY ROAD CULTURE #
#####################

class EasyRoadCulture(Culture):
    def __init__(self):
        super().__init__()
        self.name = "Easy Road Culture"
        # Properties of the culture with their default values go in self.properties.
        self.properties = {"Motorway": False,
                           "Stop Sign": False}

        self.agent_properties = {"Speed": 0}
        self.ids = {}

    def create_arguments(self):
        """
        Defines set of arguments present in the culture and their verifier functions.
        """
        args = []

        _id = 0
        motion = Argument(_id, "I will not get a ticket.")
        self.ids["no_ticket"] = _id
        motion.set_verifier(lambda gen: True)  # Propositional arguments are always valid.
        args.append(motion)

        _id += 1
        arg1 = Argument(_id, "This is a motorway.")
        self.ids["is_motorway"] = _id
        def arg1_verifier(road: RoadCell, agent: RoadAgent):
            return road["Motorway"] is True
        arg1.set_verifier(arg1_verifier)
        args.append(arg1)

        _id += 1
        arg2 = Argument(_id, "There is a stop sign.")
        self.ids["has_stop_sign"] = _id
        def arg2_verifier(road: RoadCell, agent: RoadAgent):
            return road["Stop Sign"] is True
        arg2.set_verifier(arg2_verifier)
        args.append(arg2)

        _id += 1
        speed0 = Argument(_id, "My speed is 0.")
        self.ids["speed==0"] = _id
        def speed0_verifier(road: RoadCell, agent: RoadAgent):
            return agent["Speed"] <= 0
        speed0.set_verifier(speed0_verifier)
        args.append(speed0)

        _id += 1
        speed70 = Argument(_id, "My speed is 70 or less.")
        self.ids["speed<=70"] = _id
        def speed70_verifier(road: RoadCell, agent: RoadAgent):
            return agent["Speed"] <= 70
        speed70.set_verifier(speed70_verifier)
        args.append(speed70)

        self.AF.add_arguments(args)

    def initialise_random_road(self, road: RoadCell):
        """
        Receives an empty RoadCell and initialises properties with acceptable random values.
        :param road: uninitialised RoadCell.
        """
        motorway = random.choice([True, False])
        road.assign_property_value("Motorway", motorway)

        stop_sign = random.choice([True, False])
        road.assign_property_value("Stop Sign", stop_sign)

    def define_attacks(self):
        """
        Defines attack relationships present in the culture.
        """
        ID = self.ids

        self.AF.add_attack(ID["is_motorway"], ID["no_ticket"])
        self.AF.add_attack(ID["has_stop_sign"], ID["no_ticket"])
        self.AF.add_attack(ID["has_stop_sign"], ID["is_motorway"])
        self.AF.add_attack(ID["speed==0"], ID["has_stop_sign"])
        self.AF.add_attack(ID["speed<=70"], ID["is_motorway"])



#######################
# MEDIUM ROAD CULTURE #
#######################

class MediumRoadCulture(Culture):
    def __init__(self):
        self.ids = {}
        super().__init__()
        self.name = "Hard Road Culture"
        # Properties of the culture with their default values go in self.properties.
        self.properties = {"Motorway": False,
                           "Stop Sign": False,
                           "School": False,
                           "Single Lane": False,
                           "Town Road": False
                           }

        self.agent_properties = {"Speed": 0,
                                 "Emergency Vehicle": False}

    def create_arguments(self):
        """
        Defines set of arguments present in the culture and their verifier functions.
        """
        args = []

        _id = 0
        motion = Argument(_id, "I will not get a ticket.")
        self.ids["no_ticket"] = _id
        motion.set_verifier(lambda *gen: True)  # Propositional arguments are always valid.
        args.append(motion)

        _id += 1
        arg1 = Argument(_id, "This is a motorway.")
        self.ids["is_motorway"] = _id
        def arg1_verifier(road: RoadCell, agent: RoadAgent):
            return road["Motorway"] is True
        arg1.set_verifier(arg1_verifier)
        args.append(arg1)

        _id += 1
        arg2 = Argument(_id, "There is a stop sign.")
        self.ids["has_stop_sign"] = _id
        def arg2_verifier(road: RoadCell, agent: RoadAgent):
            return road["Stop Sign"] is True
        arg2.set_verifier(arg2_verifier)
        args.append(arg2)

        _id += 1
        arg3 = Argument(_id, "There is a school nearby.")
        self.ids["has_school"] = _id
        def arg3_verifier(road: RoadCell, agent: RoadAgent):
            return road["School"] is True
        arg3.set_verifier(arg3_verifier)
        args.append(arg3)

        _id += 1
        arg4 = Argument(_id, "This is a single lane road.")
        self.ids["single_lane"] = _id
        def arg4_verifier(road: RoadCell, agent: RoadAgent):
            return road["Single Lane"] is True
        arg4.set_verifier(arg4_verifier)
        args.append(arg4)

        _id += 1
        arg5 = Argument(_id, "This is a town road.")
        self.ids["town_road"] = _id
        def arg5_verifier(road: RoadCell, agent: RoadAgent):
            return road["Town Road"] is True
        arg5.set_verifier(arg5_verifier)
        args.append(arg5)

        _id += 1
        speed0 = Argument(_id, "My speed is 0.")
        self.ids["speed==0"] = _id
        def speed0_verifier(road: RoadCell, agent: RoadAgent):
            return agent["Speed"] <= 0
        speed0.set_verifier(speed0_verifier)
        args.append(speed0)

        _id += 1
        speed20 = Argument(_id, "My speed is 20 or less.")
        self.ids["speed<=20"] = _id
        def speed20_verifier(road: RoadCell, agent: RoadAgent):
            return agent["Speed"] <= 20
        speed20.set_verifier(speed20_verifier)
        args.append(speed20)

        _id += 1
        speed30 = Argument(_id, "My speed is 30 or less.")
        self.ids["speed<=30"] = _id
        def speed30_verifier(road: RoadCell, agent: RoadAgent):
            return agent["Speed"] <= 30
        speed30.set_verifier(speed30_verifier)
        args.append(speed30)

        _id += 1
        speed60 = Argument(_id, "My speed is 60 or less.")
        self.ids["speed<=60"] = _id
        def speed60_verifier(road: RoadCell, agent: RoadAgent):
            return agent["Speed"] <= 60
        speed60.set_verifier(speed60_verifier)
        args.append(speed60)

        _id += 1
        speed70 = Argument(_id, "My speed is 70 or less.")
        self.ids["speed<=70"] = _id
        def speed70_verifier(road: RoadCell, agent: RoadAgent):
            return agent["Speed"] <= 70
        speed70.set_verifier(speed70_verifier)
        args.append(speed70)

        _id += 1
        emergency = Argument(_id, "I am an emergency vehicle.")
        self.ids["emergency_vehicle"] = _id
        def emergency_verifier(road: RoadCell, agent: RoadAgent):
            return agent["Emergency Vehicle"] is True
        emergency.set_verifier(emergency_verifier)
        args.append(emergency)

        self.AF.add_arguments(args)

    def initialise_random_road(self, road: RoadCell):
        """
        Receives an empty RoadCell and initialises properties with acceptable random values.
        :param road: uninitialised RoadCell.
        """
        motorway = random.choice([True, False])
        road.assign_property_value("Motorway", motorway)

        stop_sign = random.choice([True, False])
        road.assign_property_value("Stop Sign", stop_sign)

        school = random.choice([True, False])
        road.assign_property_value("School", school)

        single_lane = random.choice([True, False])
        road.assign_property_value("Single Lane", single_lane)

        town_road = random.choice([True, False])
        road.assign_property_value("Town Road", town_road)


    def initialise_random_agent(self, agent: RoadAgent):
        """
        Receives an empty RoadAgent and initialises properties with acceptable random values.
        :param agent: uninitialised RoadAgent.
        """
        emergency_vehicle = False if np.random.randint(0, 5) != 0 else True
        agent.assign_property_value("Emergency Vehicle", emergency_vehicle)

        speed = np.random.randint(0, 120)
        agent.assign_property_value("Speed", speed)


    def define_attacks(self):
        """
        Defines attack relationships present in the culture.
        """
        ID = self.ids

        # Base conditions for ticket.
        self.AF.add_attack(ID["is_motorway"], ID["no_ticket"])
        self.AF.add_attack(ID["has_stop_sign"], ID["no_ticket"])
        self.AF.add_attack(ID["has_school"], ID["no_ticket"])
        self.AF.add_attack(ID["single_lane"], ID["no_ticket"])
        self.AF.add_attack(ID["town_road"], ID["no_ticket"])

        # Speed checks and mitigatory rules.
        self.AF.add_attack(ID["speed==0"], ID["has_stop_sign"])
        self.AF.add_attack(ID["speed<=20"], ID["has_school"])
        self.AF.add_attack(ID["speed<=30"], ID["town_road"])
        self.AF.add_attack(ID["speed<=60"], ID["single_lane"])
        self.AF.add_attack(ID["speed<=70"], ID["is_motorway"])

        self.AF.add_attack(ID["emergency_vehicle"], ID["has_stop_sign"])
        self.AF.add_attack(ID["emergency_vehicle"], ID["has_school"])
        self.AF.add_attack(ID["emergency_vehicle"], ID["town_road"])
        self.AF.add_attack(ID["emergency_vehicle"], ID["single_lane"])
        self.AF.add_attack(ID["emergency_vehicle"], ID["is_motorway"])
