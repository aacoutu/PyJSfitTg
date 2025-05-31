"""   
This script takes as input:
    comma delimited .txt file containing times and fluorescence intensities
Output:
    browser-based real-time interactive data fitting tool for finding the glass transition temperature (Tg)
    
Install Bokeh, make sure dependencies are installed: https://docs.bokeh.org/en/latest/docs/installation.html
"""


from bokeh.layouts import column, row
from bokeh.models import CustomJS, Slider, Div, Range1d
from bokeh.plotting import ColumnDataSource, figure, output_file, show
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk
import numpy as np
import pandas as pd
import os


#TODO
###############################
Tmax = 130 # starting temperature of your data collection on cooling
Tmin = 35 # final temperature
coolRate = 1 # cooling rate (K/min)
ptsPerMin = 1 # number of data points per minute
headerSize = 4 # number of header rows in the file

analyzeDerivs = True # choose whether to analyze the first and second derivatives of the data
binSizeOne = 15 # first derivative bin size (unit is number of data points)
binSizeTwo = 15 # second derivative bin size (unit is number of data points)

analyzeMSE = True # choose whether to analyze fit line MSE for identifying initial dropoff from linearity
startBuffer = 10 # temperature difference from Tmax that yields a reasonable initial lower T bound

saveIvsT = False # choose whether to save the data reformatted with a temperature axis
###############################


# calculation for number of rows of data assuming inclusion of data at Tmax and Tmin
Trange = Tmax - Tmin
maxRows = int(Trange * ptsPerMin / coolRate) + 1

fileName = ''

# function for importing the data and converting time to temperature
def import_data():
    global fileName

    # open tkinter window
    root = Tk()
    # setting tkinter windows to appear at the front
    root.attributes("-topmost", True)
    # open file dialog
    infile = filedialog.askopenfilename(parent = root)
    # remove tkinter window
    root.destroy()
    
    # get file name, load data from file
    fileName = str(os.path.basename(infile))
    t, intensity = np.loadtxt(infile, skiprows=headerSize, max_rows=maxRows, unpack=True)
    
    # convert time (in seconds) to temperature
    T = Tmax - (t * coolRate / 60)
    
    return T, intensity

# importing the data; note that higher T data is lower index as currently formatted
T, intensity = import_data()

# saving the reformatted data as a csv file
if saveIvsT:
    dataToSave = []
    dataToSave.append(T)
    dataToSave.append(intensity)
    
    # open tkinter window
    root = Tk()
    # setting tkinter windows to appear at the front
    root.attributes("-topmost", True)
    # open file dialog
    saveFileName = filedialog.asksaveasfilename(parent = root, defaultextension = ".csv")
    # remove tkinter window
    root.destroy()
       
    pd.DataFrame(np.transpose(dataToSave)).to_csv(saveFileName, header=['T', 'I'], index=None)
    print('Saved I vs T formatted data\n')

# derivative analysis

# given the data set and some bin size, output the derivative data set via fit slopes to the bins; superior to finite differencing
def differentiate(xData, yData, binSize):
    
    # length of the derivative data set that will be output (half a bin worth of range is lost at each end, aside from one point in total)
    derivSetSize = len(xData) - binSize + 1
        
    # will hold midpoints of bins in xDeriv, best fit slopes in yDeriv
    xDeriv, yDeriv = [], []

    # iteratively fill out derivative data set arrays
    for i in range (0, derivSetSize):
        
        # set first xDeriv value to first bin midpoint position
        if i == 0:
            # (binSize - 1)/2 is the half bin lost at the start, coolRate/ptsPerMin converts number of data points to temperature
            xDeriv.append(xData[0] - ((binSize - 1)/2 * (coolRate/ptsPerMin))) 
            
        # fill out subsequent bin midpoint positions
        else:
            # coolRate/ptsPerMin is the temperature unit by which to decrement
            xDeriv.append(xDeriv[i-1] - (coolRate / ptsPerMin))
            
        # assigning the data values for the current bin
        xBin, yBin = [], []
        for j in range(i, i+binSize):
            xBin.append(xData[j])
            yBin.append(yData[j])
        
        # obtain the absolute value of the slope of the best fit line to the bin of data
        yDeriv.append(np.abs(np.polyfit(xBin, yBin, 1)[0]))

    return xDeriv, yDeriv

# will output graphs of the first and second derivatives to the console
if analyzeDerivs:

    fig = plt.figure(figsize = (8, 4))    

    # first derivatives
    xp, yp = differentiate(T, intensity, binSizeOne)
    
    ax1 = fig.add_subplot(121)
    ax1.plot(xp, yp, color = 'blue')
    ax1.set_title('1st Deriv; Bin Size: '+str(binSizeOne)+' pts')
    ax1.set(xlabel = 'T', ylabel = 'I`', xlim = (Tmin, Tmax))   
    
    # second derivatives
    xpp, ypp = differentiate(xp, yp, binSizeTwo)

    ax2 = fig.add_subplot(122)
    ax2.plot(xpp, ypp, color = 'red')
    ax2.set_title('2nd Deriv; Bin Size: '+str(binSizeTwo)+' pts')
    ax2.set(xlabel = 'T', ylabel = 'I``', xlim = (Tmin, Tmax))
    
    plt.tight_layout()
    plt.show();
    
# will output a graph of the line fit mean squared error (MSE) for data range from the starting temperature to temperature T 
if analyzeMSE:
      
    # lower temperature bounds, fit line MSE values
    lowTbound, MSE = [], []

    for i in range(0, int(len(T) - (startBuffer * (ptsPerMin / coolRate)))):
        
        # assign lower temperature bound values starting from the max minus the buffer, and decrement down in temperature units coolRate/ptsPerMin
        lowTbound.append(Tmax - startBuffer  - (i * (coolRate/ptsPerMin)))
        
        # bins to use for fitting lines to the data; range is Tmax to lowTbound
        xBin, yBin = [], []
       
        # filling in the bins with the data across the range associated with the current lowTbound
        for j in range(0, int(startBuffer * (ptsPerMin / coolRate)) + i):
            xBin.append(T[j])
            yBin.append(intensity[j])
            
        # fitting the bin of data to a line
        fit = np.polyfit(xBin, yBin, 1) 
        
        # obtain the residual sum of squares
        resSumSq = 0
        for j in range(0, len(xBin)):
            
            yTheo = fit[0]*xBin[j] + fit[1] # fit line intensity value at given x value in bin
            res = yBin[j] - yTheo # residual via comparing experimental and theoretical value
            resSumSq = resSumSq + (res*res) # summing the squares of the residuals across the bin
            
        # mean squared error (residual sum of squares divided by number of data points), normalized by intensity scale
        MSE.append(resSumSq /(len(xBin) * max(intensity)))
    
    # graphing MSE vs lower T bound
    plt.scatter(lowTbound, MSE, color = 'green')  
    plt.ylim(min(MSE), 5*min(MSE))
    plt.xlim(60, Tmax) # lower graphing bound of 60 C is convenient for our data, but change as desired
    plt.title('MSE Behavior of Line Fit from Max T')
    plt.xlabel('Lower T Fitting Bound')
    plt.ylabel('Fit Line MSE')
    plt.show();


# preparing for JavaScript callback

# axis labels
label_x = "Temperature (degrees C)"
label_y = "Intensity (a.u.)"

# controls the opacity of the lines:
alpha = 0.5
# controls the step for the slider values
step = 0.1

# graph title
title = "Tg Fitting of: "+fileName

# creating plot to display in browser
plot = figure(
   tools="box_zoom,pan,wheel_zoom,reset,save",
   x_axis_label=label_x, y_axis_label=label_y, title=title
)

# formatting data points
plot.circle(T, intensity, size=5, color="black", alpha=0.5)
data = ColumnDataSource(data=dict(x=T, y=intensity))

# formatting graph display properties
yPad = (max(intensity) - min(intensity)) / 20
xPad = (max(T) - min(T)) / 20
xLowLim = min(T) - xPad
xUpLim = max(T) + xPad
yLowLim = min(intensity) - yPad
yUpLim = max(intensity) + yPad

# glass state lower T bound
line_1_coords_x = [xLowLim, xLowLim]
line_1_coords_y = [yLowLim, yUpLim]
line_1_source = ColumnDataSource(data=dict(x=line_1_coords_x, y=line_1_coords_y))
slider_1 = Slider(start=xLowLim, end=xUpLim, value=xLowLim, step=step, title="Red start")    
plot.line('x', 'y', source=line_1_source, line_width=2, line_alpha=alpha, line_dash="dashed", color="red")

# glass state upper T bound
line_2_coords_x = [xLowLim, xLowLim] 
line_2_coords_y = [yLowLim, yUpLim]
line_2_source = ColumnDataSource(data=dict(x=line_2_coords_x, y=line_2_coords_y))
slider_2 = Slider(start=xLowLim, end=xUpLim, value=xLowLim, step=step, title="Red end")    
plot.line('x', 'y', source=line_2_source, line_width=2, line_alpha=alpha, line_dash="dashed", color="red")

# rubber state lower T bound
line_3_coords_x = [xUpLim, xUpLim] 
line_3_coords_y = [yLowLim, yUpLim]
line_3_source = ColumnDataSource(data=dict(x=line_3_coords_x, y=line_3_coords_y))
slider_3 = Slider(start=min(T), end=xUpLim, value=xUpLim, step=step, title="Blue start")    
plot.line('x', 'y', source=line_3_source, line_width=2, line_alpha=alpha, line_dash="dashed", color="blue")

# rubber state upper T bound
line_4_coords_x = [xUpLim, xUpLim] 
line_4_coords_y = [yLowLim, yUpLim]
line_4_source = ColumnDataSource(data=dict(x=line_4_coords_x, y=line_4_coords_y))
slider_4 = Slider(start=min(T), end=xUpLim, value=xUpLim, step=step, title="Blue end.")    
plot.line('x', 'y', source=line_4_source, line_width=2, line_alpha=alpha, line_dash="dashed", color="blue")

# lines of best Fit
best_fit_init_x = [xLowLim, xUpLim]
best_fit_init_y = [yLowLim, yLowLim]
best_fit_1 = ColumnDataSource(data=dict(x=best_fit_init_x, y=best_fit_init_y))
best_fit_2 = ColumnDataSource(data=dict(x=best_fit_init_x, y=best_fit_init_y))
plot.line('x', 'y', source=best_fit_1, line_width=2, line_alpha=alpha, color="red")
plot.line('x', 'y', source=best_fit_2, line_width=2, line_alpha=alpha, color="blue")

# intercept line
collision_coords_x = [xLowLim, xLowLim] 
collision_coords_y = [yLowLim, yUpLim]
collision_source = ColumnDataSource(data=dict(x=collision_coords_x, y=collision_coords_y))
plot.line('x', 'y', source=collision_source, line_width=2, line_alpha=alpha, line_dash="dashed", color="green")
plot.x_range = Range1d(xLowLim, xUpLim)
plot.y_range = Range1d(yLowLim, yUpLim)

# auto-updating text fields providing the glass transition temperature (Tg) and the fit line MSE values
Tg = Div(text = '0.0', width = 100, height = 50)
MSEglass = Div(text = '0.00000', width = 100, height = 50)
MSEliquid = Div(text = '0.00000', width = 100, height = 50)

# JavaScript callback arguments
callback_args = dict(
    coords_1 = line_1_source,
    coords_2 = line_2_source,
    coords_3 = line_3_source,
    coords_4 = line_4_source,
    coords_collision = collision_source,
    slider_1 = slider_1,
    slider_2 = slider_2,
    slider_3 = slider_3,
    slider_4 = slider_4,
    bf_1 = best_fit_1,
    bf_2 = best_fit_2,
    Tg = Tg,
    MSEglass = MSEglass,
    MSEliquid = MSEliquid,
    data=data
)


# JavaScript callback for browser-based real-time interactive data fitting
callback = CustomJS(args=callback_args,
                    code="""
    
    // for updating slider positions 
    function updateCoords(slider, line_coords){
        for (var i = 0; i < line_coords.data['x'].length; i++) {
            line_coords.data['x'][i] = slider.value 
        }
        line_coords.change.emit();
    }

    // for updating data in each interval
    function getInbetweens(data, slider_min, slider_max){
        const x_0 = slider_min.value
        const x_1 = slider_max.value
        const output_x = []
        const output_y = []
        for (var i = 0; i < data.data['x'].length; i++) {
            if (data.data['x'][i] >= x_0 && data.data['x'][i] <= x_1){
                output_x.push(data.data['x'][i])
                output_y.push(data.data['y'][i])
            }
        }
        return([output_x, output_y])
    }

    // linear least squares fitting
    function leastSqrs(values_x, values_y) {
        
        // initializing variables
        var x_sum = 0;
        var y_sum = 0;
        var xy_sum = 0;
        var xx_sum = 0;
        var count = 0;
        var x = 0;
        var y = 0;
        var values_length = values_x.length;

        if (values_length != values_y.length) {
            throw new Error('values_x and values_y must have same size');
        }

        if (values_length === 0) {
            return [ [], [] ];
        }

        // summing different variables
        for (let i = 0; i < values_length; i++) {
            x = values_x[i];
            y = values_y[i];
            x_sum += x;
            y_sum += y;
            xx_sum += x*x;
            xy_sum += x*y;
            count++;
        }

        // calculate slope m and y-intercept b
        var m = (count*xy_sum - x_sum*y_sum) / (count*xx_sum - x_sum*x_sum);
        var b = (y_sum/count) - (m*x_sum)/count;

        return [m, b]
    }

    // update the graphed fit lines whenever fitting is performed
    function linearFit(data, interval_data, bf_data){
        const coeff = leastSqrs(interval_data[0], interval_data[1])
        const c_1 = coeff[0]
        const c_0 = coeff[1]
        
        // updating lines
        for (var i = 0; i <  bf_data.data['x'].length; i++) {
            bf_data.data['y'][i] = c_1*bf_data.data['x'][i] + c_0
        }
        bf_data.change.emit()

        return [c_1, c_0]
    }
    
    // update from sliders
    updateCoords(slider_1, coords_1)
    updateCoords(slider_2, coords_2)
    updateCoords(slider_3, coords_3)
    updateCoords(slider_4, coords_4)
    const interval_1 = getInbetweens(data, slider_1, slider_2)
    const interval_2 = getInbetweens(data, slider_3, slider_4)

    // set to perform the fit whenever slider values yield a range of at least 2 data points
    var vec_blue = [0,0]
    var vec_red = [0,0]
    if (interval_1[0].length >= 2){
        vec_red = linearFit(data, interval_1, bf_1)
    }
    if (interval_2[0].length >= 2){
        vec_blue = linearFit(data, interval_2, bf_2)
    }

    // getting Tg via intersection of rubber and glass lines
    var x_collision = 0
    var y_collision = 0
    if (interval_1[0].length >= 2 && interval_2[0].length >= 2 && vec_blue[0] - vec_red[0] != 0){
        x_collision = (vec_red[1] - vec_blue[1]) / (vec_blue[0] - vec_red[0])
        y_collision = vec_red[0]*x_collision + vec_red[1]
        coords_collision.data['x'][0] = x_collision
        coords_collision.data['x'][1] = x_collision
        coords_collision.change.emit()
        Tg.text = x_collision.toFixed(1)
    }
    
    // max intensity, variables for MSE calculation
    const ymax = data.data['y'][data.data['y'].length - 1]
    var res = 0
    var res2Sum = 0
    var theoVal = 0
    var N = interval_1[0].length
    
    // auto update MSE for glass line
    for (var i = 0; i < N; i++) {
            theoVal = (vec_red[0]*interval_1[0][i]) + vec_red[1]
            res = interval_1[1][i] - theoVal
            res2Sum = res2Sum + (res*res)
    }
    MSEglass.text = (res2Sum/(N*ymax)).toFixed(8)
    
    res2Sum = 0
    N = interval_2[0].length
    
    // auto update MSE for rubber line
    for (var i = 0; i < N; i++) {
            theoVal = (vec_blue[0]*interval_2[0][i]) + vec_blue[1]
            res = interval_2[1][i] - theoVal
            res2Sum = res2Sum + (res*res)
    }
    MSEliquid.text = (res2Sum/(N*ymax)).toFixed(8)
    
    // logging
    console.clear()
    console.log("Red interval: " + [coords_1.data['x'][0], coords_2.data['x'][0]])
    console.log("Red interval best fit: c_1: " + vec_red[0] + " , c_0: " + vec_red[1])
    console.log("Blue interval: " + [coords_3.data['x'][0], coords_4.data['x'][0]])
    console.log("Blue interval best fit: c_1: " + vec_blue[0] + " , c_0: " + vec_blue[1])
    console.log("Collision point: (" + x_collision + ", " + y_collision + ")")

""")

# call JavaScript on slider change
slider_1.js_on_change('value', callback)
slider_2.js_on_change('value', callback)
slider_3.js_on_change('value', callback)
slider_4.js_on_change('value', callback)

# rendering changes
blank = Div(text = '', width = 100, height = 50)
TgTitle = Div(text = 'Tg =', width = 30, height = 50)
MSEglassTitle = Div(text = 'Normalized MSE Glass =', width = 150, height = 50)
MSEliquidTitle = Div(text = 'Normalized MSE Liquid =', width = 150, height = 50)
layout = row(plot, column(slider_1, slider_2, slider_3, slider_4, blank, row(TgTitle, Tg), row(MSEglassTitle, MSEglass), row(MSEliquidTitle, MSEliquid)))

output_file('out.html')
show(layout)
