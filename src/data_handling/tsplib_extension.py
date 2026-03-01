from typing import List

from tsplib95 import fields
from tsplib95.models import StandardProblem


class OSMIDField(fields.Field):
    """Custom field to store and parse original OSM Node IDs in TSPLib files."""

    def __init__(self, keyword: str = "OSM_IDS"):
        super().__init__(keyword)

    def parse(self, text: str) -> List[int]:
        # Converts the data in the file (space separated list) to a list of ints
        return [int(x) for x in text.strip().split()]

    def render(self, value: List[int]) -> str:
        # Converts the list of ints into a space-separated string for the file
        return " ".join(map(str, value))


class NodeLocations(fields.Field):
    """
    Custom field to store lat/long coordinates.
    Format in file:
    val1 val2
    val3 val4

    Format in Python: [(val1, val2), (val3, val4)]
    """

    def __init__(self, keyword: str = "NODE_LOCATION"):
        super().__init__(keyword)

    def parse(self, text: str) -> List[tuple]:
        # Splits the text into a flat list of floats, then groups them into pairs
        numbers = [float(x) for x in text.split()]
        return list(zip(numbers[0::2], numbers[1::2]))

    def render(self, value: List[tuple]) -> str:
        # Converts each tuple into a space-separated string and joins with newlines
        return "\n".join(f"{lat} {lng}" for lat, lng in value)


class SanitizePathField(fields.StringField):
    """
    A StringField that strips trailing whitespace and EOF markers.
    Because tusing the normal StringField it always had EOF appended to the path
    """

    def parse(self, text: str) -> str:
        # .strip() handles \n, \r, and spaces.
        # We also filter out literal 'EOF' if it's present in the text block.
        cleaned = text.strip()
        if cleaned.endswith("EOF"):
            cleaned = cleaned[:-3].strip()
        return cleaned


class TSPProblemWithOSMIDs(StandardProblem):
    """
    Extends the standard problem by storing the original OSM Node IDs in a custom field.
    Stores the path to the corresponding graph file in another custom field.
    This is usful for us, to later reconstruct the solution on the original graph and visualize it.
    """

    osm_ids = OSMIDField()
    graphml_file = SanitizePathField("GRAPHML_FILE")
    node_locations = NodeLocations()
