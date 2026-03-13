import bilby
from bilby.core.prior import Uniform
import redback
import numpy as np
import matplotlib.pyplot as plt
from alerce.core import Alerce
import pandas as pd

# Inital Alerce clience
client = Alerce()

# Let's first plot the salt2 model using the redback interface

'''
# We first set up the parameters for the model.
sncosmo_model = 'salt2' #must be the same as the sncosmo registered model name
redshift = 0.1
start_time = 55570
time = np.linspace(0.1, 65, 100) + start_time
'''

'''

# We set up a dictionary for any kwargs required by redback.
# These are similar to the kwargs required by other redback models
# e.g., an output format or bands or frequency.
kwargs = {'sncosmo_model': sncosmo_model, 'frequency': 4e14, 'output_format': 'flux_density'}

# We also set the peak time for the model in MJD
kwargs['peak_time'] = 55589.0

# We now need to pass in any extra arguments required by the model itself. For salt2 these are x0, x1, and c
model_kwargs = {'x0': 0.8, 'x1': 0.9, 'c': 0.3}
outs = redback.transient_models.supernova_models.sncosmo_models(time=time, redshift=redshift,
                                                                model_kwargs=model_kwargs, **kwargs)

# Let's plot.

plt.semilogy(time - start_time, outs, label='salt2')
plt.show()

# We can also call the model to evaluate magnitudes instead of flux densities.
kwargs = {'sncosmo_model': sncosmo_model, 'bands': 'ztfr', 'output_format': 'magnitude'}
kwargs['peak_time'] = 55589.0
outs = redback.transient_models.supernova_models.sncosmo_models(time=time, redshift=redshift,
                                                                model_kwargs=model_kwargs, **kwargs)
plt.plot(time - start_time, outs)
plt.gca().invert_yaxis()
plt.show()

# Now that we can see the model works. Let's try to fit some data.
# Let's fit some data to a ztf data
transient = 'ZTF22aalrcmn'
#data = redback.get_data.get_lasair_data(transient=transient, transient_type='supernova')
data = client.query_detections(transient, format='json', survey= 'ztf')
df = pd.DataFrame(data)

# Set up the redback transient object.
#sn = redback.transient.Supernova.from_lasair_data(transient, use_phase_model=True,data_mode='magnitude')

# Change the ZTF band filter IDs to the corresponding band names (green and red)
df['band'] = df['fid'].map({1: 'g', 2: 'r', 3: 'i'})

# Drop rows with missing values in critical columns
#df = df.dropna(subset=['mjd', 'magpsf', 'sigmapsf', 'band'])  

sn = redback.transient.Supernova(name=transient, data_mode= "magnitude", use_phase_model= False, 
                                 time=df['mjd'].to_numpy(), magnitude=df['magpsf'].to_numpy(), magnitude_err=df['sigmapsf'].to_numpy(),
                               bands=df['band'].to_numpy())

# Let's plot the data to ensure everything is set up correctly.
sn.plot_data()

# Now we set up the model and priors. For SNCOSMO model fitting, this interface is slightly different.
# To the standard in redback as we can directly sample t0 without using a phase model.
# Note you will be able get lightcurves from t0 if you just set t0 to zero.
sncosmo_model = 'salt2' #must be the same as the sncosmo registered model name

priors = bilby.core.prior.PriorDict()
priors['redshift'] = 0.061

# Set a prior on t0 to be within 100 days before the first observation
priors['t0'] = bilby.core.prior.Uniform(sn.x[0] - 100, sn.x[0] - 0.01, 't0', latex_label=r'$t_0$')

# Set a prior on the peak time to be within 10 days of the maximum (minimum magnitude)
data_peak = sn.x[np.argmin(sn.y)]
priors['peak_time'] = bilby.core.prior.Uniform(data_peak - 10, data_peak + 10, 'peak_time', latex_label=r'$t_{\rm peak}$')

# Set a prior on the x0, x1, and c parameters i.e., the salt2 model parameters
priors['x0'] = bilby.core.prior.Uniform(1e-10, 1e-1, 'x0', latex_label=r'$x_0$')
priors['x1'] = bilby.core.prior.Normal(0, 1, 'x1', latex_label=r'$x_1$')
priors['c'] = bilby.core.prior.Normal(0, 0.1, 'c', latex_label=r'$c$')

# We set up a dictionary for any kwargs required by redback.
# To make sure redback understands which keywords are model parameters, we need to pass an extra list with the names of the model parameters.
kwargs = {'sncosmo_model': sncosmo_model, 'bands': sn.filtered_sncosmo_bands, 'output_format': 'magnitude',
          'model_kwarg_names': ['x0', 'x1', 'c']}

try:

  # Let's fit. Again the interface is similar to the normal interface for redback.
  result = redback.fit_model(transient=sn, model='sncosmo_models', prior=priors, model_kwargs=kwargs,
                           sampler='ultranest', nlive=200, plot=False, outdir= f'fit_model/{transient}')
  
  ax = result.plot_lightcurve(random_models=50, show=False)
  ax.set_xscale('linear')
  ax.set_yscale('linear')
  plt.legend()
  plt.title(f'SNCOSMO Fit to {transient}')
  plt.xlabel('MJD')
  plt.ylabel('Magnitude')
  plt.show()
  
  fig = result.plot_corner()
  fig.suptitle(f'SNCOSMO Fit Corner Plot for {transient}')

except PermissionError as e:
    print(f"PermissionError: {e}. This error can occur if the output directory already exists and is not empty. Please check the 'fit_model/{transient}' directory and ensure it is empty or choose a different output directory.")
except Exception as e:
    print(f"An error occurred: {e}. Please check the error message for more details.")
'''


'''
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////

Below is an example of how to use the specific salt2 interface. 
This is a wrapper around the sncosmo_models interface but provides a more 
user-friendly interface for the salt2 model.

//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////
'''

# The above was the general SNCOSMO models interface, we also provide a specific interface for the salt2 model.
# This is similar to the interface for other redback models. e.g.,

redshift = 0.1
start_time = 55570
time = np.linspace(0.1, 65, 100) + start_time
f_ni = 0.2 #Uniform(0.001, 1, 'f_nickel', latex_label = r'$f_{\mathrm{Ni}}$')
m_ej = 1.1 #Uniform(0.01, 2, 'mej', latex_label = r'$M_{\mathrm{ej}}~(M_\odot)$')

kwargs = {'output_format': 'magnitude', 'bands':'ztfr', 'kappa': 0.05, 'kappa_gamma': 0.027, 'vej': 1e3, 'temperature_floor': 3000}
outs = redback.transient_models.supernova_models.type_1a(time=time, redshift=redshift, x0=1e-7, x1=0.9, c=0.3, peak_time=55589, f_nickel=f_ni, mej=m_ej, **kwargs)

# Where outs is just an array of magnitudes.

# You can fit with this model as you do with other redback models e.g.,

transient = 'ZTF22aalrcmn'
#data = redback.get_data.get_lasair_data(transient=transient, transient_type='supernova')
data = client.query_detections(transient, format='json')
df = pd.DataFrame(data)

# Set up the redback transient object.
#sn = redback.transient.Supernova.from_lasair_data(transient, use_phase_model=True, data_mode='magnitude')
df['band'] = df['fid'].map({1: 'g', 2: 'r', 3: 'i'})

# Drop rows with missing values in critical columns
#df = df.dropna(subset=['mjd', 'magpsf', 'sigmapsf', 'band'])  

sn = redback.transient.Supernova(name=transient, data_mode= "magnitude", use_phase_model= False, 
                                 time=df['mjd'].to_numpy(), magnitude=df['magpsf'].to_numpy(), magnitude_err=df['sigmapsf'].to_numpy(),
                               bands=df['band'].to_numpy())

priors = bilby.core.prior.PriorDict()
priors['redshift'] = 0.061

# Set a prior on t0 to be within 100 days before the first observation
priors['t0'] = bilby.core.prior.Uniform(sn.x[0] - 100, sn.x[0] - 0.01, 't0', latex_label=r'$t_0$')

# Set a prior on the peak time to be within 10 days of the maximum (minimum magnitude)
data_peak = sn.x[np.argmin(sn.y)]
priors['peak_time'] = bilby.core.prior.Uniform(data_peak - 10, data_peak + 10, 'peak_time', latex_label=r'$t_{\rm peak}$')

# Set a prior on the x0, x1, and c parameters i.e., the salt2 model parameters
priors['x0'] = bilby.core.prior.Uniform(1e-10, 1e-1, 'x0', latex_label=r'$x_0$')
priors['x1'] = bilby.core.prior.Normal(0, 1, 'x1', latex_label=r'$x_1$')
priors['c'] = bilby.core.prior.Normal(0, 0.1, 'c', latex_label=r'$c$')

# These parameters are also available as a default prior for the salt2 model.
# The access is similar to how it is for every other redback model. But please update the default prior to t0/peaktime values relevant to your data.

kwargs = {'bands': sn.filtered_sncosmo_bands, 'output_format': 'magnitude'}

try:
  # Let's fit. Again the interface is similar to the normal interface for redback.
  result = redback.fit_model(transient=sn, model='salt2', prior=priors, model_kwargs=kwargs,
                            sampler='dynesty', nlive=100, plot=False, clean=True, outdir= f'Type_Ia/{transient}')

  ax = result.plot_lightcurve(random_models=50, show=False)
  ax.set_xscale('linear')
  ax.set_yscale('linear')  
  
  fig = result.plot_corner()
  fig.suptitle(f'SNCOSMO Fit Corner Plot for {transient}')
  
except PermissionError as e:
    print(f"PermissionError: {e}. This error can occur if the output directory already exists and is not empty. Please check the 'salt2_fit_model/{transient}' directory and ensure it is empty or choose a different output directory.")
except Exception as e:
    print(f"An error occurred: {e}. Please check the error message for more details.")    