try:
	from utils.culture_lib.argument import Argument, ArgumentationFramework
except Exception as e:
    print('Warning: graph-tool not installed, using old Argumentation Framework.')
	from utils.culture_lib.argument_old import Argument, ArgumentationFramework

class Culture:
    def __init__(self):
        self.AF = ArgumentationFramework()
        self.properties = {}
        self.name = None

        self.create_arguments()
        self.define_attacks()

    def create_arguments(self):
        pass


    def define_attacks(self):
        pass
