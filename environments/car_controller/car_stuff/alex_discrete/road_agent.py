
class RoadAgent:
    def __init__(self):
        self.road_culture = None
        pass

    def __getitem__(self, item):
        return self.__dict__.get(item, None)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def culture_properties(self):
        if self.road_culture is None:
            return
        return self.road_culture.__dict__.get("agent_properties", None)

    def binary_features(self):
        features = []
        for prop in sorted(self.culture_properties().keys()):
            if prop != "Speed":
                features.append(0 if self[prop] is False else 1)
        return features

    def set_culture(self, culture):
        self.road_culture = culture
        if self.culture_properties() is None:
            print("RoadCell::set_culture: Culture {} has no properties.".format(culture.name))
            return
        for property, default_value in self.culture_properties().items():
            self.__setattr__(property, default_value)

    def assign_property_value(self, property, value):
        if hasattr(self, property) is False:
            print("RoadCell::assign_property_value: Property {} not found within road cell.".format(property))
            return
        self.__setattr__(property, value)