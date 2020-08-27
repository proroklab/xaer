from environment.car_controller.car_stuff.culture.culture import Culture
from environment.car_controller.car_stuff.culture.argument import Argument, ArgumentationFramework

class SimpleCulture(Culture):
    def __init__(self):
        # Properties of the culture with their default values go in self.properties.
        super().__init__()
        self.name = "Simple"
        # self.properties


    def create_arguments(self):
        """
        Defines set of arguments present in the culture.
        :return: Set of arguments.
        """
        args = []

        motion = Argument(0, "I can drive on this road.")
        motion.set_verifier(lambda gen: True)  # Propositional arguments are always valid.
        args.append(motion)

        rule1 = Argument(1, "Gold agents cannot drive on Grey roads.")
        def rule1_verifier(agent_colour, road_colour):
            return agent_colour == "Gold" and road_colour == "Grey"
        rule1.set_verifier(rule1_verifier)
        args.append(rule1)

        rule2 = Argument(2, "Red agents cannot drive on Purple and Brown roads.")
        def rule2_verifier(agent_colour, road_colour):
            return agent_colour == "Red" and (road_colour == "Purple" or road_colour == "Brown")
        rule2.set_verifier(rule2_verifier)
        args.append(rule2)

        rule3 = Argument(3, "Blue agents cannot drive on Purple, Olive, and Brown roads.")
        def rule3_verifier(agent_colour, road_colour):
            return agent_colour == "Blue" and (road_colour != "Grey" and road_colour != "Orange")
        rule3.set_verifier(rule3_verifier)
        args.append(rule3)
        
        rule4 = Argument(4, "Green agents cannot drive on Orange and Olive roads.")
        def rule4_verifier(agent_colour, road_colour):
            return agent_colour == "Green" and (road_colour == "Orange" or road_colour == "Olive")
        rule4.set_verifier(rule4_verifier)
        args.append(rule4)
        
        self.argumentation_framework.add_arguments(args)

    def define_attacks(self):
        """
        Defines attack relationships present in the culture.
        :return: Attack relationships.
        """
        motion_id = 0
        a1 = 1
        a2 = 2
        a3 = 3
        a4 = 4

        self.argumentation_framework.add_attack(a1, motion_id)
        self.argumentation_framework.add_attack(a2, motion_id)
        self.argumentation_framework.add_attack(a3, motion_id)
        self.argumentation_framework.add_attack(a4, motion_id)



