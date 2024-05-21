### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union, Iterable, Callable
import pathlib
import xml.etree.ElementTree as ET

### External Imports ###
import numpy as np
import pandas as pd
import torch as tc
# from aicsimageio.readers.bioformats_reader import BioFile

### Internal Imports ###

########################

# def get_physical_size_in_um(image):
#     metadata = ET.fromstring(image.ome_xml)
#     physical_size_y = 1.0
#     physical_size_x = 1.0
#     unit_x = "µm"
#     unit_y = "µm"

#     for child in metadata:
#         for inner_child in child:
#             if "PhysicalSizeX" in inner_child.attrib.keys():
#                 physical_size_x = float(inner_child.attrib["PhysicalSizeX"])
#             if "PhysicalSizeY" in inner_child.attrib.keys():
#                 physical_size_y = float(inner_child.attrib["PhysicalSizeY"])
#             if "PhysicalSizeXUnit" in inner_child.attrib.keys():
#                 unit_x = inner_child.attrib["PhysicalSizeXUnit"]
#             if "PhysicalSizeYUnit" in inner_child.attrib.keys():
#                 unit_y = inner_child.attrib["PhysicalSizeYUnit"]

#     if unit_x == "µm":
#         pass
#     elif unit_x == "mm":
#         physical_size_x = physical_size_x * 1000.0
#     elif unit_x == "nm":
#         physical_size_x = physical_size_x / 1000.0
#     else:
#         raise ValueError("Unsupported physical unit.")
    
#     if unit_y == "µm":
#         pass
#     elif unit_y == "mm":
#         physical_size_y = physical_size_y * 1000.0
#     elif unit_y == "nm":
#         physical_size_y = physical_size_y / 1000.0
#     else:
#         raise ValueError("Unsupported physical unit.")
    
#     return physical_size_y, physical_size_x, unit_x, unit_y