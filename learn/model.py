HOUSES = [
	"Ravenclaw",
	"Slytherin",
	"Gryffindor",
	"Hufflepuff",
]

def get_house_color(house) :
	if house == "Ravenclaw":
		return "blue"
	if house == "Slytherin":
		return "green"
	if house == "Gryffindor":
		return "orange"
	if house == "Hufflepuff":
		return "red"
	
DATA_MODEL = [
    {"name": "Index", "idx": 0, "type": "int"},
    {"name": "Hogwarts House", "idx": 1, "type": "string"},
    {"name": "First Name", "idx": 2, "type": "string"},
    {"name": "Last Name", "idx": 3, "type": "string"},
    {"name": "Birthday", "idx": 4, "type": "string"},
    {"name": "Best Hand", "idx": 5, "type": "string"},
    {"name": "Arithmancy", "idx": 6, "type": "float"},
    {"name": "Astronomy", "idx": 7, "type": "float"},
    {"name": "Herbology", "idx": 8, "type": "float"},
    {"name": "Defense Against the Dark Arts", "idx": 9, "type": "float"},
    {"name": "Divination", "idx": 10, "type": "float"},
    {"name": "Muggle Studies", "idx": 11, "type": "float"},
    {"name": "Ancient Runes", "idx": 12, "type": "float"},
    {"name": "History of Magic", "idx": 13, "type": "float"},
    {"name": "Transfiguration", "idx": 14, "type": "float"},
    {"name": "Potions", "idx": 15, "type": "float"},
    {"name": "Care of Magical Creatures", "idx": 16, "type": "float"},
    {"name": "Charms", "idx": 17, "type": "float"},
    {"name": "Flying", "idx": 18, "type": "float"}
]

# match models and return appropriate type from string data
def match_types(data, type) :
	if data == '':
		return None
	if type == "float":
		return float(data)
	if type == "int":
		return int(data)
	return data


class Model:
	def __init__(self):
		self.features = {}

	def __repr__(self):
		return f"{self.features.__repr__()}\n"

	def __str__(self):
		return self.features.__str__()

	def set_feature(self, feature, value):
		self.features[feature] = value

	def get_feature(self, feature):
		try:
			return self.features[feature]
		except:
			return None

	def get_all_features(self):
		return self.features