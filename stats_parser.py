import sys
import json
import numpy as np

file = sys.argv[1]
stat = 'episode_reward_mean'

line_iter = open(file,'r')
running = True
exp_stats_dict = {}
stats_dict = {}
while running:
    line = next(line_iter, None)
    if line is None:
        running = False
        break
    if line.find("Extracting data from:") > -1:
        version = line.replace("Extracting data from: ",'').strip()
        line = next(line_iter)
        while stat not in line:
            line = next(line_iter)
        stats_json = '{'
        line = next(line_iter)
        while '##########' not in line:
            stats_json += ' ' + line
            line = next(line_iter)
        # print(stats_json)
        stats = json.loads(stats_json)
        ####
        median_list = [round(float(x['median']),2) for x in stats['quantiles']]
        lower_quartile_list = [round(float(x['lower_quartile']),2) for x in stats['quantiles']]
        upper_quartile_list = [round(float(x['upper_quartile']),2) for x in stats['quantiles']]
        best_median_idx = np.argmax(median_list)
        stats_dict[version] = f"{median_list[best_median_idx]} ({lower_quartile_list[best_median_idx]} - {upper_quartile_list[best_median_idx]})"
    if line.find('Plot figure saved in  ') > -1:
        experiment = line.replace('Plot figure saved in  ','').replace('.png','').strip()
        exp_stats_dict[experiment] = stats_dict
        stats_dict = {}
print(stat)
print(json.dumps(exp_stats_dict, indent=4))
