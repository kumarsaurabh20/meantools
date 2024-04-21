import svgutils.transform as sg
from svgelements import *

class Svg(object):
	"svg files with a data object (the svg), width, height and coordinates"

	def __init__(self, data, dim, coords):
		self.data = data
		self.width = dim[0]
		self.height = dim[1]
		self.x = coords[0]
		self.y = coords[1]

	def scale_width_to_reference(self, reference_width):
		"""Proportionally scale the image to a given width."""
		scalings_factor = reference_width / self.width
		self.data.moveto(0, 0, scale_x=scalings_factor)
		self.width = self.width * scalings_factor
		self.height = self.height * scalings_factor

	def scale_by_factor(self, scalings_factor):
		"""Proportionally scale image by a scaling factor."""
		self.data.moveto(0, 0, scale_x=scalings_factor)
		self.width = self.width * scalings_factor
		self.height = self.height * scalings_factor

	def move(self, x, y):
		"""Move the coordinates of an image."""
		self.data.moveto(x, y)
		self.x = x
		self.y = y


def get_size(svg_file):
	"""Naively parse the svg text file to get the width and height."""
	svg = SVG.parse(svg_file)
	width = svg[0].width
	height = svg[0].height
	return int(width), int(height)


def rescale(svgs):
	"""Change the dimensions of the images to the desired combinations."""
	file_list = list(svgs)
	reference = list(svgs)[0]
	for e in file_list:
		if e == reference:
			pass
		else:
			svgs[e].scale_width_to_reference(svgs[reference].width)

def change_positions(svgs):
	"""Move the images to the desired positions."""
	file_list = list(svgs)
	reference = list(svgs)[0]
	length = svgs[reference].height
	for e in file_list:
		if e == reference:
			pass
		else:
			svgs[e].move(0, 0 + length)
			length += svgs[e].height

def letter_annotations(svgs):
	"""Add letters based on the location of the images."""
	return [sg.TextElement(value.x + 10, value.y + 15, key, size=15, weight="bold") for key, value in svgs.items()]


def files_to_svg_dict(files):
	"""Convert a list of images to a dictionary.
	Mapping the image basename to the Svg class instance,
	setting the dimensions based on sizes and coordinates (0,0) by default
	"""
	return {s.split('.')[0]: Svg(data=sg.fromfile(s).getroot(), dim=get_size(s), coords=(0, 0)) for s in files}


def main():
	svgs = files_to_svg_dict(["NP_27475.svg", "VM_9056664.svg", "VM_3107397.svg", "VM_8404030.svg", "VM_9695712.svg", "VM_9084708.svg"])
	file_list = list(svgs)
	reference = list(svgs)[0]
	rescale(svgs)
	change_positions(svgs)
	full_width = svgs[reference].width
	full_height = sum([svgs[i].height for i in file_list])
	fig = sg.SVGFigure(full_width, full_height)
	text = letter_annotations(svgs)
	fig.append([s.data for s in svgs.values()])
	fig.append(text)
	fig.save("combined.svg")
