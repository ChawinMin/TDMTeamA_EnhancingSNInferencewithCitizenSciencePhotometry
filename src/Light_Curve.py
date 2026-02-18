from alerce.core import Alerce
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd

#Initialize Alerce client
client = Alerce()

#Query light curve data for a specific object
lightCurve = client.query_detections('ZTF20abjonjs', format='json')
df = pd.DataFrame(lightCurve)

#Set font properties to serif and figure size
plt.rcParams['font.family'] = 'serif'
plt.figure(figsize=(10, 8))

#Separate data by filter ID (fid) to green and red
greens = df['fid'] == 1
reds = df['fid'] == 2

#Plot light curve based on mjd and magnitude and the error bars
plt.scatter(df.loc[greens, 'mjd'], df.loc[greens, 'magpsf'], c='green', label='g-band', s= 25, alpha=0.9)
plt.errorbar(df.loc[greens, 'mjd'], df.loc[greens, 'magpsf'],yerr=df.loc[greens, 'sigmapsf'], color='green', ls='', capsize=3, capthick=1)
plt.scatter(df.loc[reds, 'mjd'], df.loc[reds, 'magpsf'], c='red', label='r-band', s= 25, alpha=0.9)
plt.errorbar(df.loc[reds, 'mjd'], df.loc[reds, 'magpsf'],yerr=df.loc[reds,'sigmapsf'] ,color='red', ls='', capsize=3, capthick=1)

#Invert y-axis for magnitude
plt.gca().invert_yaxis()

#Customize ticks spacing
plt.gca().xaxis.set_major_locator(MultipleLocator(25))
plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
plt.gca().yaxis.set_minor_locator(MultipleLocator(0.1))

#Customize tick parameters
plt.gca().tick_params(which='both', direction="in", top=True, right=True)
plt.gca().tick_params(which='major', length=6, width = 1.0)
plt.gca().tick_params(which='minor', length=3, width = 0.8)

#Set tick label font size
plt.gca().tick_params(axis='x', labelsize=14)
plt.gca().tick_params(axis='y', labelsize=14)

#Add labels and legend
plt.title('Light Curve for ZTF20abjonjs', fontsize=16)
plt.xlabel('MJD', fontsize=12)
plt.ylabel('Magnitude', fontsize=14)
plt.legend()
plt.show()