from utils.culture_lib.argument import ArgumentationFramework
from utils.culture_lib.argument_old import ArgumentationFramework_2

class Culture:
    def __init__(self):
        self.AF = ArgumentationFramework()
        self.argumentation_framework = ArgumentationFramework_2()
        self.properties = {}
        self.name = None

        self.create_arguments()
        self.define_attacks()

    def create_arguments(self):
        pass


    def define_attacks(self):
        pass
