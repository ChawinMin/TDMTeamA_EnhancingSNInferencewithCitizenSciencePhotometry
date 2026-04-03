import bilby
from bilby.core.prior import Uniform
import redback
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from alerce.core import Alerce
import pandas as pd
import redback
from redback.model_library import all_models_dict
from redback.likelihoods import GaussianLikelihoodQuadratureNoise
from redback import filters
from astropy.io import ascii
import astropy.units as u

# Inital Alerce clience
client = Alerce()


def _sanitize_latex_label(label):
    """Replace Unicode-only labels with ASCII-safe mathtext."""
    if not isinstance(label, str):
        return label

    if label in {"σ", "Ïƒ"}:
        return r"$\sigma$"

    return label


def _sanitize_result_labels(result):
    if getattr(result, "priors", None):
        for prior in result.priors.values():
            if hasattr(prior, "latex_label"):
                prior.latex_label = _sanitize_latex_label(prior.latex_label)

    if hasattr(result, "parameter_labels"):
        result.parameter_labels = [
            _sanitize_latex_label(label) for label in result.parameter_labels
        ]

    return result


def _save_corner_plot(result, filename, title):
    result = _sanitize_result_labels(result)
    with plt.rc_context({"text.usetex": False}):
        fig = result.plot_corner(save=False)
        fig.suptitle(title)
        fig.savefig(filename, dpi=200, bbox_inches="tight")
    return fig


def _save_lightcurve_plot(result, filename, title, random_models=50):
    transient = result.transient
    x_min = float(np.min(transient.x))
    x_max = float(np.max(transient.x))
    x_span = max(x_max - x_min, 1.0)
    x_padding = max(0.1 * x_span, 2.0)

    fig, ax = plt.subplots()
    ax = result.plot_lightcurve(
        axes=ax,
        random_models=random_models,
        xlim_low=x_min - x_padding,
        xlim_high=x_max + x_padding,
        xlabel="Time [MJD]",
        show=False,
        save=False
    )
    ax.set_title(title)
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    return ax

model = 'type_1a'

redshift = 0.1
start_time = 55570
time = np.linspace(0.1, 65, 100) + start_time

# Use fixed numeric values when evaluating the model directly.
mej = 1.1
f_nickel = 0.2
vej = 1e4
kappa = 0.10

kwargs = {
    'output_format': 'magnitude',
    'bands': 'ztfr',
    'kappa_gamma': 0.027,
    'temperature_floor': 3000
}
outs = redback.transient_models.supernova_models.type_1a(
    time=time, redshift=redshift, x0=1e-7, x1=0.9, c=0.3, peak_time=55589,
    f_nickel=f_nickel, mej=mej, vej=vej, kappa=kappa, **kwargs)

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
#priors['t0'] = bilby.core.prior.Uniform(sn.x[0] - 100, sn.x[0] - 0.01, 't0', latex_label=r'$t_0$')

# True explosion date
priors['t0'] = bilby.core.prior.Uniform(
                                        minimum     = df['mjd'].values.min() - 200,
                                        maximum     = df['mjd'].values.min() - 1,
                                        name        = 't0',
                                        latex_label = r'$t_{\rm expl.}~\rm (day)$',
                                        unit        = None,
                                        boundary    = None
                                        )

# Extinction
priors['av_host'] = bilby.core.prior.Uniform(
                                        minimum     = 0,
                                        maximum     = 1,
                                        name        = 'av_host',
                                        latex_label = r'$A_V~\rm (mag)$',
                                        unit        = None,
                                        boundary    = None
                                        )

# Set a prior on the peak time to be within 10 days of the maximum (minimum magnitude)
data_peak = sn.x[np.argmin(sn.y)]
priors['peak_time'] = bilby.core.prior.Uniform(data_peak - 10, data_peak + 10, 'peak_time', latex_label=r'$t_{\rm peak}$')

# Set a prior on the x0, x1, and c parameters i.e., the type_Ia model parameters
priors['mej'] = Uniform(0.01, 2, 'mej', latex_label=r'$M_{\mathrm{ej}}~(M_\odot)$')
priors['f_nickel'] = Uniform(0.001, 1, 'f_nickel', latex_label=r'$f_{\mathrm{Ni}}$')
priors['vej'] = Uniform(1e3, 40e3, 'vej', latex_label=r'$v_{\mathrm{ej}}~(\mathrm{km}/\mathrm{s})$')
priors['kappa'] = Uniform(0.05, 0.15, 'kappa', latex_label=r'$\kappa~(\mathrm{cm}^{2}/\mathrm{g})$')

# White noise parameter. Accounts for parts of data that model can't explain
priors['sigma'] = bilby.core.prior.Uniform(
                                        minimum     = 0.001,
                                        maximum     = 2,
                                        name        = 'sigma',
                                        latex_label = r'$\sigma$',
                                        unit        = None,
                                        boundary    = None
                                        )

function = all_models_dict[model]

# Likelihood function that takes into account the white noise parameter
likelihood_func = GaussianLikelihoodQuadratureNoise(
                                        x        = sn.x,
                                        y        = sn.y,
                                        function = function,
                                        kwargs   = kwargs,
                                        sigma_i  = sn.y_err
                                        )

# Keep only fixed, non-sampled settings in kwargs.
kwargs = {
    'bands': sn.filtered_sncosmo_bands,
    'output_format': 'magnitude',
    'kappa_gamma': 0.027,
    'temperature_floor': 3000
}

try:
    outdir = Path(f'supernova/type_1a/{transient}')
    result_path = outdir / f'{transient}_result.json'

    # Reuse an existing completed result when only plotting failed.
    if result_path.exists():
        result = redback.result.read_in_result(filename=str(result_path))
    else:
        result = redback.fit_model(
            transient=sn,
            model=model,
            prior=priors,
            model_kwargs=kwargs,
            sampler='dynesty',
            nlive=100,
            plot=False,
            clean=True,
            outdir=str(outdir)
        )

    fig = _save_corner_plot(
        result=result,
        filename=outdir / f'{transient}_corner.png',
        title=f'SNEIa Fit Corner Plot for {transient}'
    )
    ax = _save_lightcurve_plot(
        result=result,
        filename=outdir / f'{transient}_lightcurve.png',
        title=f'SNEIa Fit Lightcurve for {transient}',
        random_models=50
    )
    plt.show()
  
except PermissionError as e:
    print(f"PermissionError: {e}. This error can occur if the output directory already exists and is not empty. Please check the 'Type_Ia/{transient}' directory and ensure it is empty or choose a different output directory.")
except Exception as e:
    print(f"An error occurred: {e}. Please check the error message for more details.")    
