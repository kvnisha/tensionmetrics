"""
Script for reading chains and quantifying tensions between datasets for DESI-Y1 BAO-FS

Metrics covered:
Chi-squared                             DONE
Bayes factor                            DONE (requires anesthetic)
Deviance Information Criterion (DIC)    DONE
1D Marginalized Difference              Sort of
Updated Difference in Mean (QUDM)
Index of Inconsistency (IOI)
Suspiciousness

"""
import warnings
import logging

import numpy as np
import scipy 
import getdist

import tensiometer
from tensiometer import utilities
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
        
        self.samples_type = self.sample1_type
        

    def compute_deltachi2(self):
        r"""
        Compute delta chi2 between datasets: chi2 = 2*[-log[Like]]

        Assume the first `getdist.MCSamples` instance in `self.samples` is the fiducial/baseline
        Try to load chi2 from `.minimum` or `.bestfit` file
        If no such file found, use -log(Like).max() in `getdist.MCSamples`
        """
        MAP_chi2s = []
        for sample in self.samples:
            try:
                #getdist.MCSamples.getBestFit().logLike returns bestfit -log(Like)
                MAP_minusloglike = sample.getBestFit().logLike
            except FileNotFoundError as err:
                warnings.warn(f"Cannot load `BestFit` from file. No `.minimum` or `.bestfit` file found. Use estimate of MAP from chain.", UserWarning)
                #getdist.MCSamples.getLikeStats().logLike_sample returns MCMC bestfit -log(Like)
                MAP_minusloglike = sample.getLikeStats().logLike_sample
            MAP_chi2s.append(2*MAP_minusloglike)
        MAP_chi2s = np.asarray(MAP_chi2s)
        return MAP_chi2s - MAP_chi2s[0]
    
    def compute_deltak(self):
        r"""
        Compute the difference between the degrees of freedom between datasets.
        Assume the first `getdist.MCSamples` instance in `self.samples` is the fiducial/baseline
        """
        ks = []
        for sample in self.samples:
            ks.append(len(sample.getParamSampleDict(0, want_derived=False, want_fixed=False)))
        ks = np.asarray(ks)
        return ks - ks[0]
    
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
        for sample in self.samples:
            try:
                pDIC = -2.0 * (sample.getBestFit().logLike - sample.getLikeStats().meanLogLike)
                DIC = 2.0 * (sample.getBestFit().logLike + pDIC)
            except FileNotFoundError as err:
                warnings.warn(f"Cannot load `BestFit` from file. No `.minimum` or `.bestfit` file found. Use estimate of MAP from chain.", UserWarning)                    
                pDIC = 2.0 * (sample.getLikeStats().varLogLike)
                DIC = 2.0 * (sample.getLikeStats().logLike_sample + pDIC)
            DICs.append(DIC)
        DICs = np.asarray(DICs)
        return DICs - DICs[0]
    
    def convert_chi2_to_probability(self):
        r"""
        Compute the probability to exceed given the chi2 values.
        Assume the first `getdist.MCSamples` instance in `self.samples` is the fiducial/baseline
        """
        p = []
        if not (np.all(self.compute_deltachi2>0)):
            raise ValueError('Chi2 must be positive\n')
        if not (np.all(self.compute_deltak>0)):
            raise ValueError('Degrees of freedom must be positive\n')
        p = scipy.stats.chi2.cdf(self.compute_deltachi2(), self.compute_deltak())
        return p
    
    def convert_probability_to_nsigma(self):
        r"""
        Convert the probability to exceed to effective number of sigmas.
        nsigma (P) = sqrt(2) * erf^{-1}(P)
        Assume the first `getdist.MCSamples` instance in `self.samples` is the fiducial/baseline
        Also implemented as a tensiometer function tensiometer.utilities.from_confidence_to_sigma()
        """

    def convert_nsigma_to_probability(self):
        r"""
        Convert the effective number of sigmas to probability to exceed.
        Assume the first `getdist.MCSamples` instance in `self.samples` is the fiducial/baseline
        Also implemented as a tensiometer function tensiometer.utilities.from_sigma_to_confidence()
        !Check for the asymptotic form!
        """
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
        params1 = tensionmetrics.give_param_names(self.samples[0])
        params2 = tensionmetrics.give_param_names(self.samples[1])
        common_params=[name for name in params1 if name in params2]
        if len(common_params) == 0:
            raise ValueError("No common parameters found between datasets.")
        return common_params        

    def compute_1d_marginalized_difference(self, parameter):
        r"""
        Compute the one-dimensional marginalized parameter differences between datasets.
        
        Parameters
        ----------

        parameter: str
            Name of the parameter to compute the difference of

        Returns 1d marginalized parameter difference, PTE, n-sigma
        """
        if parameter not in tensionmetrics.check_common_param(self):
            raise ValueError(f"Parameter {parameter} not found in both datasets.")
        # else:
            
        
        # Compute differences
        # nsigma = {}
        # for param in parameters:
            # Extract the bestfit value of the parameter from both samples
        params1 = self.sample1.paramNames.parWithName(param).bestfit_sample
        p2 = self.sample2.paramNames.parWithName(param).bestfit_sample
        #sigma_p1 = 
            #sigma_p2 = 
            # Compute the difference
            #nsigma[param] = (p1-p2)/(np.sqrt(sigma_p1**2 + sigma_p2**2))
        return p1, p2

# OKAY BAYES FACTOR WORKS!
    def compute_bayes_factor(self):
        r"""
        Compute the Bayes factor between datasets.
        Bayes_factor = logR = LogZ1 - LogZ2
        
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
                

# def probability_to_nsigma(P):
#     """
#     nsigma (P) = sqrt(2) * erf^{-1}(P)

#     Also implemented as a tensiometer function tensiometer.utilities.from_confidence_to_sigma()
#     """
#     if (np.all(P<0) or np.all(P>1)):
#         raise ValueError('Probability must be between 0 and 1\n')
#     return np.sqrt(2)*scipy.special.erfinv(P)


# def nsigma_to_probability(nsigma):
#     """
#     Returns the probability to exceed the given number of sigmas

#     Also implemented as a tensiometer function tensiometer.utilities.from_sigma_to_confidence()
#     !Check for the asymptotic form!
#     """
#     if (np.all(nsigma<0)):
#         raise ValueError('Number of sigmas must be positive\n')
    
#     return scipy.special.erf(nsigma/np.sqrt(2))

# def chi2_to_nsigma(chi2, ndof):
#     """
#     Returns the number of sigmas for a chi2 

#     Also implemented as a tensiometer function tensiometer.utilities.from_chi2_to_sigma()
#     """
#     if (np.all(chi2<0)):
#         raise ValueError('Chi2 must be positive\n')
#     if (np.all(ndof<0)):
#         raise ValueError('Degrees of freedom must be positive\n')
#     p = scipy.stats.chi2.cdf(chi2, ndof)
#     return probability_to_nsigma(p)

# def bayes_factor(getdist_samples):
#     """
#     Input: getdist.mcsamples.MCSamples instances 
    
#     """
#     return

# def suspiciousness(path_to_chain1, path_to_chain2):
#     """
#     Input: getdist.mcsamples.MCSamples instances
    
#     """
#     return

# def deviance_information_criterion(path_to_chain1, path_to_chain2):
#     """
#     Input: getdist.mcsamples.MCSamples instances
    
#     """
#     return

# def QUDM(path_to_chain1, path_to_chain2):
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
