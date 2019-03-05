import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy
import imageio
import tensorflow as tf
import math
from tqdm import tqdm
from numba import guvectorize, float64, complex64, int32, float32


def mandelbrot(coords, maxiters=100, smoothing=True, bailout=2):
    val = np.zeros(coords.shape)
    z = np.zeros(coords.shape, np.complex64)
    c = coords.copy()
    thres = bailout**2
    
    for it in range(1, maxiters+1):
        notdone = np.less(z.real*z.real + z.imag*z.imag, thres)
        val[notdone] = it
        z[notdone] = z[notdone]**2 + c[notdone]
        
    if smoothing:
        il = 1 / np.log(2)
        lp = np.log(np.log(bailout))
        val = 0.05 * (val + il*lp - il*np.log(np.log(np.abs(z))))
        
    return val


def mandelbrot_gpu(coords, maxiters=100, bailout=2):
    maxiters_array = np.ones(coords.shape, int) * maxiters
    bailout_array = np.ones(coords.shape, int) * bailout
    return mandelbrot_gpu_func(coords, maxiters_array, bailout_array)


@guvectorize([(complex64[:], int32[:], int32[:], float32[:])], '(n),(n),(n)->(n)', target='cuda')
def mandelbrot_gpu_func(coords, maxiters_array, bailout_array, output):
    maxiters = maxiters_array[0]
    bailout = bailout_array[0]
    threshold = bailout * bailout
    for i in range(coords.shape[0]):
        creal = coords[i].real
        cimag = coords[i].imag
        zreal = 0
        zimag = 0
        output[i] = np.nan
        for it in range(maxiters):
            zreal2 = zreal*zreal
            zimag2 = zimag*zimag
            if zreal2 + zimag2 > threshold:
                il = 1 / math.log(float(2))
                lp = math.log(math.log(float(bailout)))
                output[i] = 0.05 * (it + il*lp - il*math.log(math.log(math.sqrt(zreal2+zimag2))))
                break
            zimag = 2 * zreal*zimag + cimag
            zreal = zreal2 - zimag2 + creal


def julia(coords, c, maxiters=100, smoothing=True, bailout=2):
    val = np.zeros(coords.shape)
    z = coords.copy()
    thres = bailout**2
    
    for it in range(1, maxiters+1):
        notdone = np.less(z.real*z.real + z.imag*z.imag, thres)
        val[notdone] = it
        z[notdone] = z[notdone]**2 + c
        
    if smoothing:
        il = 1 / np.log(2)
        lp = np.log(np.log(bailout))
        val = 0.05 * (val + il*lp - il*np.log(np.log(np.abs(z))))
    
    return val


def julia_gpu(coords, c, maxiters=100, bailout=2):
    maxiters_array = np.ones(coords.shape, int) * maxiters
    bailout_array = np.ones(coords.shape, int) * bailout
    c_array = np.ones(coords.shape, np.complex64) * c
    return julia_gpu_func(coords, c_array, maxiters_array, bailout_array)


@guvectorize([(complex64[:], complex64[:], int32[:], int32[:], float32[:])], '(n),(n),(n),(n)->(n)', target='cuda')
def julia_gpu_func(coords, c_array, maxiters_array, bailout_array, output):
    maxiters = maxiters_array[0]
    bailout = bailout_array[0]
    threshold = bailout * bailout
    creal = c_array[0].real
    cimag = c_array[0].imag
    for i in range(coords.shape[0]):
        zreal = coords[i].real
        zimag = coords[i].imag
        output[i] = np.nan
        for it in range(maxiters):
            zreal2 = zreal*zreal
            zimag2 = zimag*zimag
            if zreal2 + zimag2 > threshold:
                il = 1 / math.log(float(2))
                lp = math.log(math.log(float(bailout)))
                output[i] = 0.05 * (it + il*lp - il*math.log(math.log(math.sqrt(zreal2+zimag2))))
                break
            zimag = 2 * zreal*zimag + cimag
            zreal = zreal2 - zimag2 + creal

def create_set(center, view, aspect=1.5, res=600):
    xmin, xmax = (center[0]-view*aspect, center[0]+view*aspect)
    ymin, ymax = (center[1]-view, center[1]+view)
    xdist = abs(xmin) + abs(xmax)
    ydist = abs(ymin) + abs(ymax)

    coords = np.mgrid[xmin:xmax:res*aspect*1j, ymin:ymax:res*1j]
    coords = coords[0] + coords[1]*1j
    
    return coords.astype(np.complex64)


def transform(val, density=1, shift=0):
    val *= density
    val += shift
    val %= 1
    val = np.log(val+1)

    return val


def cmap_colorize(val, cmap='viridis'):
    nan_idx = np.isnan(val)
    val = cm.get_cmap(cmap)(val)
    val[nan_idx] = [0,0,0,1]
    val = (val*255).round().astype(np.uint8)

    return val


def colorize(val, colormap='mandelbrot'):
    if colormap == 'mandelbrot':
        positions = np.array([0.0, 0.16, 0.42, 0.6425, 0.8575, 1.0])
        colors = np.array([
            [0, 7, 100],
            [32, 107, 203],
            [237, 255, 255],
            [255, 170, 0],
            [0, 2, 0],
            [0, 7, 100],
        ])
    elif colormap == 'julia':
        positions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        colors = np.array([
            [0, 0, 133],
            [255, 255, 245],
            [255, 181, 0],
            [156, 0, 0],
            [0, 0, 133],
        ])
    colors = colors.T

    R = scipy.interpolate.PchipInterpolator(positions, colors[0])
    G = scipy.interpolate.PchipInterpolator(positions, colors[1])
    B = scipy.interpolate.PchipInterpolator(positions, colors[2])
    
#     pos_list = np.linspace(0, 1, 101)
#     R_list = []
#     G_list = []
#     B_list = []
#     for i in pos_list:
#         R_list.append(R(i))
#         G_list.append(G(i))
#         B_list.append(B(i))
#     plt.figure(figsize=(18, 3))
#     plt.plot(pos_list, R_list, color='r')
#     plt.plot(pos_list, G_list, color='g')
#     plt.plot(pos_list, B_list, color='b')
#     plt.show()
    
    x = val.reshape(val.shape + (1,))
    
#     sns.distplot(x[~np.isnan(x)])
#     plt.show()
    
    px = np.concatenate((R(x), G(x), B(x)), axis=2)
    px[np.isnan(px)] = 0
    return px.round().astype(np.uint8)


def draw(val, name):
    plt.figure(figsize=(18, 12))
    fig = plt.imshow(val, interpolation='lanczos')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.show()
    imageio.imsave('media/'+name+'.png', val)
#     plt.imsave(name+'.png', val)


def julia_frame(args):
    coords, a = args
    # c = 0.7885 * np.exp(a*1j)
    c = 0.5 * np.exp(a*1j)
    val = julia_gpu(coords, c, maxiters=500, bailout=128).T
    val = transform(val, density=0.25)
    # val = colorize(val, colormap='mandelbrot')
    val = cmap_colorize(val, cmap='jet')
    return val