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

# flags = SimpleNamespace(**{
# 	"gif_speed": 0.25, # "GIF frame speed in seconds."
# 	"max_plot_size": 20, # "Maximum number of points in the plot. The smaller it is, the less RAM is required. If the log file has more than max_plot_size points, then max_plot_size means of slices are used instead."
# })
linestyle_set = ['-', '--', '-.', ':', '']
color_set = list(mcolors.TABLEAU_COLORS)

def wrap_string(s, max_len=10):
	return '\n'.join([
		s[i*max_len:(i+1)*max_len]
		for i in range(int(np.ceil(len(s)/max_len)))
	]).strip()

def line_plot(logs, figure_file, max_plot_size=20, show_deviation=False, base_list=None, base_shared_name='baseline', average_non_baselines=None):
	assert not base_list or len(base_list)==len(logs), f"base_list (len {len(base_list)}) and logs (len {len(logs)}) must have same lenght or base_list should be empty"
	log_count = len(logs)
	# Get plot types
	stats = [None]*log_count
	key_ids = {}
	for i in range(log_count):
		log = logs[i]
		# Get statistics keys
		if log["length"] < 2:
			continue
		(step, obj) = log["line_example"]
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
	lines_dict = {}
	for log_id in range(log_count):
		log = logs[log_id]
		name = log["name"]
		data_iter = log["data_iter"]
		length = log["length"]
		if length < 2:
			print(name, " has not enough data for a reasonable plot")
			continue
		print('Extracting data from:',name)
		if length > max_plot_size:
			plot_size = max_plot_size
			data_per_plotpoint = length//plot_size
		else:
			plot_size = length
			data_per_plotpoint = 1
		# Build x, y
		stat = stats[log_id]
		x = {
			key:[]
			for key in stat
		}
		y = {
			key:{
				"min":float("+inf"), 
				"max":float("-inf"), 
				"quantiles":[]
			}
			for key in stat
		}
		last_step = 0
		for _ in range(plot_size):
			# initialize
			values = {
				key: []
				for key in stat
			}
			# compute values foreach key
			plotpoint_i = 0
			for (step, obj) in data_iter:
				if step <= last_step:
					continue
				plotpoint_i += 1
				last_step = step
				for key in stat: # foreach statistic
					v = obj.get(key,None)
					if v is not None:
						values[key].append(v)
				if plotpoint_i > data_per_plotpoint: # save plotpoint
					break
			# add average to data for plotting
			for key in stat: # foreach statistic
				value_list = values[key]
				if len(value_list) <= 0:
					continue
				stats_dict = y[key]
				stats_dict["quantiles"].append([
					np.quantile(value_list,0.25) if show_deviation else 0, # lower quartile
					np.quantile(value_list,0.5), # median
					np.quantile(value_list,0.75) if show_deviation else 0, # upper quartile
				])
				stats_dict["min"] = min(stats_dict["min"], min(value_list))
				stats_dict["max"] = max(stats_dict["max"], max(value_list))
				x[key].append(last_step)
		lines_dict[name] = {
			'x': x,
			'y': y,
			'log_id': log_id
		}
	plotted_baseline = False
	plot_dict = {}
	for name, line in lines_dict.items():
		is_baseline = base_list and name in base_list
		if plotted_baseline and is_baseline:
			continue # already plotted
		if is_baseline:
			name = base_shared_name
		# Populate axes
		print('#'*20)
		print(name)
		x = line['x']
		y = line['y']
		log_id = line['log_id']
		stat = stats[log_id]
		plot_list = []
		for j in range(ncols):
			for i in range(nrows):
				idx = j if nrows == 1 else i*ncols+j
				if idx >= len(stat):
					continue
				key = stat[idx]
				y_key = y[key]
				x_key = x[key]
				y_key_lower_quartile, y_key_median, y_key_upper_quartile = map(np.array, zip(*y_key["quantiles"]))
				if base_list:
					base_line = base_list[log_id]
					base_y_key = lines_dict[base_line]['y'][key]
					base_y_key_lower_quartile, base_y_key_median, base_y_key_upper_quartile = map(np.array, zip(*base_y_key["quantiles"]))
					normalise = lambda x,y: 100*(x-y)/(y-base_y_key['min']+1)
					y_key_median = normalise(y_key_median, base_y_key_median)
					y_key_lower_quartile = normalise(y_key_lower_quartile, base_y_key_lower_quartile)
					y_key_upper_quartile = normalise(y_key_upper_quartile, base_y_key_upper_quartile)
				# print stats
				print(f"    {key} is in [{y_key['min']},{y_key['max']}] with medians: {y_key_median}")				
				if is_baseline:
					plotted_baseline = True
				plot_list.append({
					'coord': (i,j), 
					'key': key,
					'x': x_key,
					'y_q1': y_key_lower_quartile,
					'y_q2': y_key_median, 
					'y_q3': y_key_upper_quartile
				})
		plot_dict[name] = plot_list

	##############################
	##### Merge non-baselines ####
	if average_non_baselines:
		avg_fn = np.mean if average_non_baselines=='mean' else np.median
		new_plot_dict = {}
		merged_plots = {
			'coord': [],
			'key': [],
			'x': [],
			'y_q1': [],
			'y_q2': [], 
			'y_q3': []
		}
		for name, plot_list in plot_dict.items():
			is_baseline = base_list and name == base_shared_name
			if is_baseline:
				new_plot_dict[name] = plot_list
				continue
			merged_plots['coord'].append([plot['coord'] for plot in plot_list])
			merged_plots['key'].append([plot['key'] for plot in plot_list])
			merged_plots['x'].append([plot['x'] for plot in plot_list])
			merged_plots['y_q1'].append([plot['y_q1'] for plot in plot_list])
			merged_plots['y_q2'].append([plot['y_q2'] for plot in plot_list])
			merged_plots['y_q3'].append([plot['y_q3'] for plot in plot_list])
		new_plot_dict['XARL'] = [
			{
				'coord': coord,
				'key': key,
				'x': x,
				'y_q1': y_q1,
				'y_q2': y_q2,
				'y_q3': y_q3
			}
			for coord, key, x, y_q1, y_q2, y_q3 in zip(
				merged_plots['coord'][0],
				merged_plots['key'][0],
				merged_plots['x'][0],
				avg_fn(merged_plots['y_q1'], axis=0),
				avg_fn(merged_plots['y_q2'], axis=0),
				avg_fn(merged_plots['y_q3'], axis=0),
			)
		]
		plot_dict = new_plot_dict
	###############################	

	for log_id, (name, plot_list) in enumerate(plot_dict.items()):
		for plot in plot_list:
			i,j = plot['coord']
			x_key = plot['x']
			key = plot['key']
			y_key_lower_quartile = plot['y_q1']
			y_key_median = plot['y_q2']
			y_key_upper_quartile = plot['y_q3']
			# ax
			ax_id = key_ids[key]
			ax = axes[ax_id]
			format_label = lambda x: x.replace('_',' ')
			ax.set_ylabel(wrap_string(format_label(key) if not base_list else f'{format_label(key)} - % of gain over baseline', 25), fontdict=font_dict)
			ax.set_xlabel('step', fontdict=font_dict)
			# plot mean line
			ax.plot(x_key, y_key_median, label=format_label(name), linestyle=linestyle_set[log_id//len(color_set)], color=color_set[log_id%len(color_set)])
			# plot std range
			if show_deviation:
				ax.fill_between(x_key, y_key_lower_quartile, y_key_upper_quartile, alpha=0.25, color=color_set[log_id%len(color_set)])
			# show legend
			ax.legend()
			# display grid
			ax.grid(True)

	figure.savefig(figure_file,bbox_inches='tight')
	print("Plot figure saved in ", figure_file)
	figure = None

def line_plot_files(url_list, name_list, figure_file, max_length=None, max_plot_size=20, show_deviation=False, base_list=None, base_shared_name='baseline', average_non_baselines=None, statistics_list=None):
	assert len(url_list)==len(name_list), f"url_list (len {len(url_list)}) and name_list (len {len(name_list)}) must have same lenght"
	logs = []
	for url,name in zip(url_list,name_list):
		length, line_example = get_length_and_line_example(url)
		if max_length:
			length = max_length
		print(f"{name} has length {length}")
		logs.append({
			'name': name, 
			'data_iter': parse(url, max_i=length, statistics_list=statistics_list), 
			'length':length, 
			'line_example': parse_line(line_example, statistics_list=statistics_list)
		})
	line_plot(logs, figure_file, max_plot_size, show_deviation, base_list, base_shared_name, average_non_baselines)
		
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

def parse_line(line, i=0, statistics_list=None):
	val_dict = json.loads(line)
	step = val_dict["info"]["num_steps_sampled"] # "num_steps_sampled", "num_steps_trained"
	# obj = {
	# 	"median cum. reward": np.median(val_dict["hist_stats"]["episode_reward"]),
	# 	"mean visited roads": val_dict['custom_metrics'].get('visited_junctions_mean',val_dict['custom_metrics'].get('visited_cells_mean',0))
	# }
	obj = {
		"episode_reward_median": np.median(val_dict["hist_stats"]["episode_reward"]),
	}
	for k in ["episode_reward_mean","episode_reward_max","episode_reward_min","episode_len_mean"]:
		obj[k] = val_dict[k] 
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
	if statistics_list:
		statistics_list = set(statistics_list)
		obj = dict(filter(lambda x: x[0] in statistics_list, obj.items()))
	return (step, obj)
	
def parse(log_fname, max_i=None, statistics_list=None):
	with open(log_fname, 'r') as logfile:
		for i, line in enumerate(logfile):
			if max_i and i > max_i:
				return
			try:
				yield parse_line(line, i=i, statistics_list=statistics_list)
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
	
def make_gif(gif_path, file_list, gif_speed=0.25):
	with imageio_get_writer(gif_path, mode='I', duration=gif_speed) as writer:
		for filename in file_list:
			image = imageio_imread(filename)
			writer.append_data(image)