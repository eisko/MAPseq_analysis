# create custum colormaps
import numpy as np
# for custom colormaps
from matplotlib.colors import ListedColormap

# green = OMC
# purple = ACA
# blue = Lab Mouse
# orange = Singing mouse

N = 256
purple = np.ones((N, 4))
purple[:, 0] = np.linspace(1, 146/256, N) # R = 146
purple[:, 1] = np.linspace(1, 39/256, N) # G = 39
purple[:, 2] = np.linspace(1, 143/256, N)  # B = 143
purple_cmp = ListedColormap(purple)

green = np.ones((N, 4))
green[:, 0] = np.linspace(1, 5/256, N) # R = 5
green[:, 1] = np.linspace(1, 104/256, N) # G = 104
green[:, 2] = np.linspace(1, 57/256, N)  # B = 57
green_cmp = ListedColormap(green)

orange = np.ones((N, 4))
orange[:, 0] = np.linspace(1, 242/256, N) # R = 242
orange[:, 1] = np.linspace(1, 101/256, N) # G = 101
orange[:, 2] = np.linspace(1, 34/256, N)  # B = 34
orange_cmp = ListedColormap(orange)

blue = np.ones((N, 4))
blue[:, 0] = np.linspace(1, 33/256, N) # R = 33
blue[:, 1] = np.linspace(1, 64/256, N) # G = 64
blue[:, 2] = np.linspace(1, 154/256, N)  # B = 154
blue_cmp = ListedColormap(blue)