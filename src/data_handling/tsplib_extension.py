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


class TSPProblemWithOSMIDs(StandardProblem):
    """
    Extends the standard problem by storing the original OSM Node IDs in a custom field.
    Stores the path to the corresponding graph file in another custom field.
    This is usful for us, to later reconstruct the solution on the original graph and visualize it.
    """

    osm_ids = OSMIDField()
    graphml_file = fields.StringField("GRAPHML_FILE")
