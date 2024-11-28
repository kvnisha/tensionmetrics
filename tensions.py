"""
Script for reading chains and quantifying tensions between datasets

Metrics covered:

Model comparison:
    Chi-squared                             
    Bayes factor                        
    Suspiciousness!
    Akaike Information Criterion (AIC)    
    Deviance Information Criterion (DIC)

Internal/ External Consistency:
    1D Marginalized Difference
    Difference in Mean (DM)!
    QMAP!              
    Updated Difference in Mean (QUDM)
    Index of Inconsistency (IOI)!


"""

import warnings
import logging

import numpy as np
import scipy 
import getdist

import tensiometer
# from tensiometer import utilities
from tensiometer import gaussian_tension

import anesthetic


class tensionmetrics:
    r"""
    A class to compute tension metrics between datasets 
    """
    def __init__(self, sample1, sample2):
        r"""
        tensionmetrics class constructor
        to initialize the attributes of the class

        Parameters
        ----------
        sample1 & sample2: getdist.mcsamples.MCsamples instances OR anesthetic.samples.NestedSamples instances
            sample1 is assumed to be baseline/fiducial model.
        """
        self.sample1 = sample1
        self.sample2 = sample2

        self.Logger = logging.getLogger(__name__)
        
        # Ensure both samples are provided
        if sample1 is None or sample2 is None:
            raise TypeError(f"Please input two `getdist.mcsamples.MCSamples` OR `anesthetic.samples.NestedSamples` instances.")
        self.sample1_type = type(sample1)
        self.sample2_type = type(sample2)

        # Check if the samples are cobaya mcmc chains or polychord (anesthetic) chains
        if isinstance(sample1, getdist.mcsamples.MCSamples) and isinstance(sample2, getdist.mcsamples.MCSamples):
            self.Logger.info(f"Two `getdist.mcsamples.MCSamples` instances provided.")
        elif isinstance(sample1, anesthetic.samples.NestedSamples) and isinstance(sample2, anesthetic.samples.NestedSamples):
            self.Logger.info(f"Two `anesthetic.samples.NestedSamples` instances provided.")
        else:
            raise TypeError(f"Please input two `getdist.mcsamples.MCSamples` OR `anesthetic.samples.NestedSamples` instances.")


    def get_chi2(sample):
        r"""
        Extract chi2 from the sample

        chi-square = - 2 * log(Like)
        """
        try:
            MAP_minusloglike = sample.getBestFit().logLike
        except getdist.mcsamples.MCSamplesError as err:
            error_message = str(err)
            if "Best fit can only be included if loaded from file and file_root.minimum exists (cannot be calculated from samples)" in error_message:
                warnings.warn("Cannot load `BestFit` from file. No `.minimum` or `.bestfit` file found. Use estimate of MAP from chain.", UserWarning)
                MAP_minusloglike = sample.getLikeStats().logLike_sample
        chi2 = 2*MAP_minusloglike
        return chi2

    def get_dof(sample):
        r"""
        Extract degrees of freedom from the sample
        """
        k = len(sample.getParamSampleDict(0, want_derived=False, want_fixed=False))
        return k

    def compute_deltachi2(self):
        r"""
        Compute delta chi2 between datasets: chi2 = 2*[-log[Like]]

        Assume the first `getdist.MCSamples` instance in `self.samples` is the fiducial/baseline
        """
        MAP_chi2s = []
        for sample in (self.sample1, self.sample2):
            MAP_chi2s.append(tensionmetrics.get_chi2(sample))
        MAP_chi2s = np.asarray(MAP_chi2s)
        return (MAP_chi2s[1] - MAP_chi2s[0])


    def compute_deltak(self):
        r"""
        Compute the difference between the degrees of freedom between datasets.
        Assume the first `getdist.MCSamples` instance in `self.samples` is the fiducial/baseline
        """
        ks = []
        for sample in (self.sample1, self.sample2):
            ks.append(tensionmetrics.get_dof(sample))
        ks = np.asarray(ks)
        return (ks[1] - ks[0])


    def compute_deltachi2_over_deltak(self):
        r"""
        Compute delta chi2 / delta k between datasets.
        Assume the first `getdist.MCSamples` instance in `self.samples` is the fiducial/baseline
        """
        deltachi2 = self.compute_deltachi2()
        deltak = self.compute_deltak()
        return deltachi2 / deltak

    
    def compute_deltaAIC(self):
        r"""
        Compute delta Akaike Information Criterion (AIC) between datasets.
        AIC = 2[-log(Like)] + 2k
        Reference: (https://arxiv.org/abs/2207.05766), Eq.(F5)
        In practice, delta AIC = delta_chi2 + 2*delta_k 
        """
        return self.compute_deltachi2() + 2.0*self.compute_deltak()


    def compute_deltaDIC(self):
        r"""
        Compute delta Deviance Information Criterion (DIC) between datasets.
        DIC = 2[-log(Like)] - 2*pDIC where pDIC = 2[logLike] - 2*<logLike>
        Reference: (https://arxiv.org/abs/2207.05766), Eq.(F6-F7)
        In practice, delta DIC = -delta_chi2 - 2*delta <logLike>
        """
        DICs = []
        for sample in (self.sample1, self.sample2):
            try:
                pDIC = -2.0 * (sample.getBestFit().logLike - sample.getLikeStats().meanLogLike)
                DIC = 2.0 * (sample.getBestFit().logLike + pDIC)
            except getdist.mcsamples.MCSamplesError as err:
                error_message = str(err)
                if "Best fit can only be included if loaded from file and file_root.minimum exists (cannot be calculated from samples)" in error_message:
                    warnings.warn("Cannot load `BestFit` from file. No `.minimum` or `.bestfit` file found. Use estimate of MAP from chain.", UserWarning)
                    pDIC = 2.0 * (sample.getLikeStats().varLogLike)
                    DIC = 2.0 * (sample.getLikeStats().logLike_sample + pDIC)
            DICs.append(DIC)
        DICs = np.asarray(DICs)
        return (DICs[1] - DICs[0])
    
    def convert_probability_to_nsigma(p):
        r"""
        Convert the probability to exceed to effective number of sigmas.
        nsigma (P) = sqrt(2) * erf^{-1}(P)
        Also implemented as a tensiometer function tensiometer.utilities.from_confidence_to_sigma()
        """
        if (np.all(p<0) or np.all(p>1)):
            raise ValueError('Probability must be between 0 and 1\n')
        return np.sqrt(2.)*scipy.special.erfinv(p)


    def convert_nsigma_to_probability(nsigma):
        r"""
        Convert the effective number of sigmas to probability to exceed.
        P = erf{ nsigma(P)/sqrt(2) }
        Also implemented as a tensiometer function tensiometer.utilities.from_sigma_to_confidence()
        """
        if (np.all(nsigma<0)):
            raise ValueError('Number of sigmas must be positive\n')
        return scipy.special.erf(nsigma/np.sqrt(2.))


    def convert_chi2_to_probability(chi2, df):
        r"""
        Compute the probability to exceed given the chi2, df values.
        !Check for the asymptotic form!
        Also implemented as a tensiometer function tensiometer.utilities.from_chi2_to_sigma()
        
        """
        if not (np.all(chi2>0)):
            raise ValueError('Chi2 must be positive\n')
        if not (np.all(df>0)):
            raise ValueError('Degrees of freedom must be positive\n')
        return scipy.stats.chi2.cdf(chi2, df)    


    def convert_chi2_to_nsigma(chi2, df):
        r"""
        Compute the probability to exceed given the chi2, df values.
        !Check for the asymptotic form!
        Also implemented as a tensiometer function tensiometer.utilities.from_chi2_to_sigma()
        
        """
        if not (np.all(chi2>0)):
            raise ValueError('Chi2 must be positive\n')
        if not (np.all(df>0)):
            raise ValueError('Degrees of freedom must be positive\n')
        p = tensionmetrics.convert_chi2_to_probability(chi2, df)
        return tensionmetrics.convert_probability_to_nsigma(p)


    def give_param_names(sample):
        r"""
        Returns the list of parameters in the given dataset
        """
        params = []
        if isinstance(sample, getdist.mcsamples.MCSamples):
            params = sample.getParamNames().list()
        elif isinstance(sample, anesthetic.samples.NestedSamples):
            params = list(sample.columns.levels[0])
        else:
            raise TypeError("Not a `getdist.mcsamples.MCSamples` or `anesthetic.samples.NestedSamples` instance.")    
        return params

    
    def check_common_param(self):
        r"""
        Check for common parameters between datasets
        
        Parameters
        ----------
        sample1 & sample2: `getdist.mcsamples.MCSamples` or `anesthetic.samples.NestedSamples` instances
        
        Returns
        List of common parameters
        """
        params1 = tensionmetrics.give_param_names(self.sample1)
        params2 = tensionmetrics.give_param_names(self.sample2)
        common_params=[name for name in params1 if name in params2]
        if len(common_params) == 0:
            raise ValueError("No common parameters found between datasets.")
        return common_params    
      

    def compute_1d_marginalized_difference(self, param):
        r"""
        Compute the one-dimensional marginalized parameter differences between the posterior means of datasets.
        
        Parameters
        ----------

        parameter: str
            Name of the parameter to compute the difference of

        Returns 1d marginalized parameters, nsigma
        """
        if param not in tensionmetrics.check_common_param(self):
            raise ValueError(f"Parameter {param} not found in both datasets.")         
        
        try:
            p1 = self.sample1.getMargeStats().parWithName(param).mean
            p2 = self.sample2.getMargeStats().parWithName(param).mean
            sigma_p1 = self.sample1.getMargeStats().parWithName(param).err
            sigma_p2 = self.sample2.getMargeStats().parWithName(param).err
        # I need to check what error is obtained when margestat is not available!
        except FileNotFoundError as err:
            warnings.warn(f"Cannot load `MargeStats` from file. No `.margestats` file found. Using MargeStats from mcmc chain.", UserWarning)
            p1 = self.sample1.paramNames.parWithName(param).mean
            p2 = self.sample2.paramNames.parWithName(param).mean
            sigma_p1 = self.sample1.paramNames.parWithName(param).err
            sigma_p2 = self.sample2.paramNames.parWithName(param).err
        nsigma = np.abs((p1-p2))/np.sqrt(sigma_p1**2 + sigma_p2**2) 
        return p1, p2, nsigma


    def compute_bayes_factor(self):
        r"""
        Compute the Bayes factor between datasets.
        Bayes_factor : logR = LogZ1 - LogZ2
        
        Requires: anesthetic

        Parameters
        ----------
        sample1 & sample2: anesthetic.samples.NestedSamples instances
            sample1 is assumed to be baseline/fiducial model.
        """
        if not (isinstance(self.sample1, anesthetic.samples.NestedSamples) and isinstance(self.sample2, anesthetic.samples.NestedSamples)):
            raise TypeError(f"Please input two `anesthetic.samples.NestedSamples` instances.")
        else:
            logZ1 = self.sample1.stats().logZ
            logZ2 = self.sample2.stats().logZ
            return logZ2 - logZ1
    
    def compute_QDM(self):
        r"""
        Compute the Difference in Mean (Q_DM) between two guassian datasets.
        
        Requires: tensiometer

        Q_DM = (mu1 - mu2) C_DM^{-1} (mu1 - mu2)^T
        C_DM = C1 + C2 - C1.C_PI^{-1}.C2 - C2.C_PI^{-1}.C1

        This is chi-2 distributed for dofs ~ rank(C_DM)^{-1}

        Parameters
        ----------
        sample1 & sample2: getdist.MCSamples instances
            sample1 is assumed to be baseline/fiducial model.     

        Returns
        -------
        Q_DM, dofs   
        """
        Q_DM, Q_DM_dofs = gaussian_tension.Q_DM(self.sample1, self.sample2)
        return Q_DM, Q_DM_dofs

    def compute_QUDM(self):
        r"""
        Compute the Updated Difference in Mean (Q_DM) between two guassian datasets.

        Requires: tensiometer

        Q_UDM = (mu1 - mu12)^T C_UDM^{-1} (mu1 - mu12)
        C_UDM = C1 - C12

        Parameters
        ----------
        sample1 & sample2: getdist.MCSamples instances
            sample1 is assumed to be baseline/fiducial model.     

        Returns
        -------
        Q_UDM, dofs   
        """
        Q_UDM, Q_UDM_dofs = gaussian_tension.Q_UDM(self.sample1, self.sample2)
        return Q_UDM, Q_UDM_dofs


    


#############################################################################
                


# def suspiciousness(path_to_chain1, path_to_chain2):
#     """
#     Input: getdist.mcsamples.MCSamples instances
    
#     """
#     return


# def IOI(path_to_chain1, path_to_chain2):
#     """
#     Input: getdist.mcsamples.MCSamples instances
    
#     """
#     return

# def print_latex_table(chains, params):
#     """
#     Input: getdist.mcsamples.MCSamples instances, params
    
#     """
#     return
