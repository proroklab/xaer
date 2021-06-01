# -*- coding: utf-8 -*-
import xarl.utils.plot_lib as plt
import argparse

parser = argparse.ArgumentParser(description='draw plots')
parser.add_argument('-o', '--output_file', dest='output_file', type=str, help='the file in which to save the plots')
parser.add_argument('-u', '--input_file_url', dest='input_file_urls', type=str, action='append', help='log files used to build the plot')
parser.add_argument('-n', '--input_file_name', dest='input_file_names', type=str, action='append', default=[], help='the name of the files used to build the plot')
parser.add_argument('-b', '--input_file_base', dest='input_file_bases', type=str, action='append', default=[], help='the base used for normalising the line plot against a specified X ax')
parser.add_argument('-l', '--max_length', dest='max_length', type=int, default=None)
parser.add_argument('-p', '--max_plot_size', dest='max_plot_size', type=int, default=20, help="Maximum number of points in the plot. The smaller it is, the less RAM is required. If the log file has more than max_plot_size points, then max_plot_size means of slices are used instead.")
parser.add_argument('-s', '--visible_statistic', dest='statistics_list', type=str, action='append', default=[], help='the name of the statistics to plot, if not all')
parser.add_argument('--show_deviation', dest='show_deviation', action='store_true')
parser.add_argument('--base_shared_name', dest='base_shared_name', type=str, default='baseline')
parser.add_argument('--average_non_baselines', dest='average_non_baselines', type=str, default=None)
parser.set_defaults(show_deviation=False)
ARGS = parser.parse_args()
print("ARGS:", ARGS)

plt.line_plot_files(
	url_list=ARGS.input_file_urls, 
	name_list=ARGS.input_file_names if ARGS.input_file_names else ARGS.input_file_urls, 
	figure_file=ARGS.output_file,
	max_length=ARGS.max_length,
	max_plot_size=ARGS.max_plot_size,
	show_deviation=ARGS.show_deviation,
	base_list=ARGS.input_file_bases,
	base_shared_name=ARGS.base_shared_name,
	average_non_baselines=ARGS.average_non_baselines,
	statistics_list=ARGS.statistics_list,
)