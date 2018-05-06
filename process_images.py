# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 10:35:04 2018

@author: bickels
"""

import os
import re
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from cell_detection.cell_detection import process_image

filename_regex = re.compile(r'(?:(\d+|(?:\d+-\d+))) (\d+x)_(\d+-\d+)_.*')
ceramic_regex = re.compile(r'([a-zA-Z]+)\((.+) (.+)\)')


def treat_ceramic_samples(ceramic_type):
    ceramic_path = os.path.join(path, ceramic_type)
    images = glob(ceramic_path + "/*.tif")
    image_names = map(lambda x: x.groups(), filter(lambda x: x is not None, map(filename_regex.match, images)))
    psm, psi, nut = ceramic_regex.match(ceramic_type).groups()

    def process_image_to_dataframe(grain, mag, repl):
        filename = "{0} {1}_{2}".format(grain, mag, repl)
        _, mask, blobs, ncells, area, dxy, dist = process_image(path, filename, mag)
        y, x, sigma = blobs.T
        return pd.DataFrame({'psm': psm,
                             'psi': psi,
                             'nutrients': nut[:-1],
                             'replicate': repl,
                             'grain size': grain,
                             'magnification': mag,
                             'dxy': dxy,
                             'davg_k1': np.mean(dist[:, 1]),
                             'dstd_k1': np.std(dist[:, 1]),
                             'ncells': ncells,
                             'area': area,
                             'mean size': np.mean(2 ** 0.5 * sigma * dxy),
                             'std size': np.std(2 ** 0.5 * sigma * dxy),
                             'filename': filename
                             })

    return pd.concat(map(process_image_to_dataframe, image_names))


def treat_day_experiments(experiment_folder, save=False, output_file=None):
    if output_file is None:
        output_file = os.path.join(path, 'results.xlsx')
    day = os.path.dirname(experiment_folder)
    sheet_name = "Day{0}".format(day)
    res = pd.concat(map(treat_ceramic_samples, os.walk(path)))
    res = res.reset_index()
    res['cell density'] = res['ncells'] / res['area']
    if save:
        res.to_excel(output_file, sheet_name=sheet_name)
    return res


def visualize(results, save=False, output_path=None):
    if save and output_path is None:
        raise ValueError("Output cannot be None if saving figures.")
    sns.set_style("ticks")

    plot = sns.factorplot(x='psi', y='cell density', hue='nutrients', row='magnification', data=results)
    plt.show()
    if save:
        plot.savefig(os.path.join(output_path, 'magnification.png'))

    plot = sns.boxplot(x='psi', y='cell density', hue='magnification', data=results)
    plt.show()

    plot = sns.boxplot(x='psi', y='cell density', hue='grain size', data=results)
    # plot.savefig(os.path.join(output_path, 'hydration.png'))
    plt.show()

    plot = sns.factorplot(x='psi', y='davg_k1', hue='grain size', data=results)
    plt.show()
    if save:
        plot.savefig(os.path.join(output_path, 'distance.png'))


if __name__ == "__main__":
    path = r'/Users/jingyuwang/Desktop/Hydration/day2'
    output_file = path + "/results.xlsx"
    results = treat_day_experiments(path, save=True, output_file=output_file)
    visualize(results, save=True, output_path=path)
