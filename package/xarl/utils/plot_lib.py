# -*- coding: utf-8 -*-
import math
import re
from types import SimpleNamespace

from PIL import Image, ImageFont, ImageDraw  # images
from imageio import get_writer as imageio_get_writer, imread as imageio_imread  # GIFs
from matplotlib import rc as matplotlib_rc # for regulating font
from matplotlib import use as matplotlib_use
matplotlib_use('Agg',force=True) # no display
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from scipy.interpolate import interp1d
from seaborn import heatmap as seaborn_heatmap  # Heatmap

import numpy as np
import json
import numbers

font_dict = {'size':22}
# matplotlib_rc('font', **font_dict)

flags = SimpleNamespace(**{
	"gif_speed": 0.25, # "GIF frame speed in seconds."
	"max_plot_size": 10, # "Maximum number of points in the plot. The smaller it is, the less RAM is required. If the log file has more than max_plot_size points, then max_plot_size means of slices are used instead."
})
linestyle_set = ['-', '--', '-.', ':', '']
color_set = list(mcolors.TABLEAU_COLORS)

def wrap_string(s, max_len=10):
	return '\n'.join([
		s[i*max_len:(i+1)*max_len]
		for i in range(int(np.ceil(len(s)/max_len)))
	]).strip()

def plot(logs, figure_file):
	log_count = len(logs)
	# Get plot types
	stats = [None]*log_count
	key_ids = {}
	for i in range(log_count):
		log = logs[i]
		# Get statistics keys
		if log["length"] < 2:
			continue
		(step, obj) = parse_line(log["line_example"])
		log_keys = list(obj.keys()) # statistics keys sorted by name
		for key in log_keys:
			if key not in key_ids:
				key_ids[key] = len(key_ids)
		stats[i] = log_keys
	max_stats_count = len(key_ids)
	if max_stats_count <= 0:
		print("Not enough data for a reasonable plot")
		return
	# Create new figure and two subplots, sharing both axes
	ncols=3 if max_stats_count >= 3 else max_stats_count
	nrows=math.ceil(max_stats_count/ncols)
	# First set up the figure and the axis
	# fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10)) # this method causes memory leaks
	figure = Figure(figsize=(10*ncols,7*nrows))
	canvas = FigureCanvas(figure)
	grid = GridSpec(ncols=ncols, nrows=nrows)
	axes = [figure.add_subplot(grid[id//ncols, id%ncols]) for id in range(max_stats_count)]
	# Populate axes
	for log_id in range(log_count):
		log = logs[log_id]
		name = log["name"]#[:10]
		data = log["data"]
		length = log["length"]
		if length < 2:
			print(name, " has not enough data for a reasonable plot")
			continue
		if length > flags.max_plot_size:
			plot_size = flags.max_plot_size
			data_per_plotpoint = length//plot_size
		else:
			plot_size = length
			data_per_plotpoint = 1
		# Build x, y
		x = {}
		y = {}
		stat = stats[log_id]
		for key in stat: # foreach statistic
			y[key] = {"min":float("+inf"), "max":float("-inf"), "data":[], "std":[]}
			x[key] = []
		last_step = 0
		for _ in range(plot_size):
			values = {}
			# initialize
			for key in stat: # foreach statistic
				values[key] = []
			# compute values foreach key
			plotpoint_i = 0
			for (step, obj) in data:
				plotpoint_i += 1
				if step <= last_step:
					continue
				last_step = step
				for key in stat: # foreach statistic
					if key not in obj:
						continue
					v = obj[key]
					values[key].append(v)
					if v > y[key]["max"]:
						y[key]["max"] = v
					if v < y[key]["min"]:
						y[key]["min"] = v
				if plotpoint_i > data_per_plotpoint: # save plotpoint
					break
			# add average to data for plotting
			for key in stat: # foreach statistic
				if len(values[key]) > 0:
					y[key]["data"].append(np.mean(values[key]))
					y[key]["std"].append(np.std(values[key]))
					x[key].append(last_step)
		# Populate axes
		print(name)
		for j in range(ncols):
			for i in range(nrows):
				idx = j if nrows == 1 else i*ncols+j
				if idx >= len(stat):
					continue
				key = stat[idx]
				ax_id = key_ids[key]
				ax = axes[ax_id]
				y_key = y[key]
				x_key = x[key]
				# print stats
				print("    ", y_key["min"], " < ", key, " < ", y_key["max"])
				# ax
				ax.set_ylabel(wrap_string(key, 20), fontdict=font_dict)
				ax.set_xlabel('step', fontdict=font_dict)
				# ax.plot(x, y, linewidth=linewidth, markersize=markersize)
				y_key_mean = np.array(y_key["data"])
				y_key_std = np.array(y_key["std"])
				#===============================================================
				# # build interpolators
				# mean_interpolator = interp1d(x_key, y_key_mean, kind='linear')
				# min_interpolator = interp1d(x_key, y_key_mean-y_key_std, kind='linear')
				# max_interpolator = interp1d(x_key, y_key_mean+y_key_std, kind='linear')
				# xnew = np.linspace(x_key[0], x_key[-1], num=plot_size, endpoint=True)
				# # plot mean line
				# ax.plot(xnew, mean_interpolator(xnew), label=name)
				#===============================================================
				# plot mean line
				ax.plot(x_key, y_key_mean, label=name, linestyle=linestyle_set[log_id//len(color_set)], color=color_set[log_id%len(color_set)])
				# plot std range
				ax.fill_between(x_key, y_key_mean-y_key_std, y_key_mean+y_key_std, alpha=0.25, color=color_set[log_id%len(color_set)])
				# show legend
				ax.legend()
				# display grid
				ax.grid(True)
	figure.savefig(figure_file,bbox_inches='tight')
	print("Plot figure saved in ", figure_file)
	figure = None

def plot_files(url_list, name_list, figure_file, max_length=None):
	logs = []
	for url,name in zip(url_list,name_list):
		length, line_example = get_length_and_line_example(url)
		if max_length:
			length = max_length
		print(f"{name} has lenght {length}")
		logs.append({'name': name, 'data': parse(url, length), 'length':length, 'line_example':line_example})
	plot(logs, figure_file)
		
def get_length_and_line_example(file):
	try:
		with open(file, 'r') as lines_generator:
			tot = 1
			line_example = next(lines_generator)
			for line in lines_generator:
				tot += 1
				if len(line) > len(line_example):
					line_example = line
			return tot, line_example
	except:
		return 0, None

def parse_line(line,i=0):
	val_dict = json.loads(line)
	step = val_dict["info"]["num_steps_sampled"] # "num_steps_sampled", "num_steps_trained"
	obj = {
		k:val_dict[k] 
		for k in ["episode_reward_mean","episode_reward_max","episode_reward_min","episode_len_mean"]
	}
	default_learner = val_dict["info"]["learner"]["default_policy"]
	obj.update({
		k:v 
		for k,v in default_learner.items()
		if isinstance(v, numbers.Number)
	})
	if 'buffer' in val_dict:
		default_buffer = val_dict["buffer"]["default_policy"]
		if 'cluster_capacity' in default_buffer:
			obj.update({
				f'capacity_{k}':v 
				for k,v in default_buffer['cluster_capacity'].items()
				if isinstance(v, numbers.Number)
			})
		if 'cluster_priority' in default_buffer:
			obj.update({
				f'priority_{k}':v 
				for k,v in default_buffer['cluster_priority'].items()
				if isinstance(v, numbers.Number)
			})
	if "custom_metrics" in val_dict:
		obj.update({
			f'env_{k}':v 
			for k,v in val_dict['custom_metrics'].items()
			if isinstance(v, numbers.Number)
		})
	return (step, obj)
	
def parse(log_fname, max_i=None):
	with open(log_fname, 'r') as logfile:
		for i, line in enumerate(logfile):
			if max_i and i > max_i:
				return
			try:
				yield parse_line(line,i)
			except Exception as e:
				print("exc %s on line %s" % (repr(e), i+1))
				print("skipping line")
				continue
	
def heatmap(heatmap, figure_file):
	# fig, ax = matplotlib.pyplot.subplots(nrows=1, ncols=1, sharey=False, sharex=False, figsize=(10,10)) # this method causes memory leaks
	figure = Figure()
	canvas = FigureCanvas(figure)
	ax = figure.add_subplot(111) # nrows=1, ncols=1, index=1
	seaborn_heatmap(data=heatmap, ax=ax)
	figure.savefig(figure_file,bbox_inches='tight')
	
def ascii_image(string, file_name):
	# find image size
	font = ImageFont.load_default()
	splitlines = string.splitlines()
	text_width = 0
	text_height = 0
	for line in splitlines:
		text_size = font.getsize(line) # for efficiency's sake, split only on the first newline, discard the rest
		text_width = max(text_width,text_size[0])
		text_height += text_size[1]+5
	text_width += 10
	# create image
	source_img = Image.new('RGB', (text_width,text_height), "black")
	draw = ImageDraw.Draw(source_img)
	draw.text((5, 5), string, font=font)
	source_img.save(file_name, "JPEG")
	
def combine_images(images_list, file_name):
	imgs = [ Image.open(i) for i in images_list ]
	# pick the smallest image, and resize the others to match it (can be arbitrary image shape here)
	min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
	imgs_comb = np.hstack( [np.asarray( i.resize(min_shape) ) for i in imgs] )
	# save the picture
	imgs_comb = Image.fromarray( imgs_comb )
	imgs_comb.save( file_name )
	
def rgb_array_image(array, file_name):
	img = Image.fromarray(array, 'RGB')
	img.save(file_name)
	
def make_gif(gif_path, file_list):
	with imageio_get_writer(gif_path, mode='I', duration=flags.gif_speed) as writer:
		for filename in file_list:
			image = imageio_imread(filename)
			writer.append_data(image)