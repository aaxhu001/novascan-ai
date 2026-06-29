import lightkurve as lk
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
import astropy.units as u
import warnings
import pywt
import time
from scipy.stats import entropy
from astropy.timeseries import LombScargle

warnings.filterwarnings('ignore')

def preprocess_lightcurve(lc):
    """
    Robust Preprocessing & Denoising Pipeline.
    """
    # 1. Remove NaNs and extreme outliers (Sigma clipping)
    lc = lc.remove_nans().remove_outliers(sigma=5.0)
    
    # 2. Wavelet Denoising & Normalization (normalize AFTER denoising)
    try:
        flux = lc.flux.value
        wavelet = 'db4'
        level = 3
        
        # Decompose
        coeffs = pywt.wavedec(flux, wavelet, level=level)
        
        # Threshold each level
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(flux)))
        
        coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        
        # Reconstruct
        flux_clean = pywt.waverec(coeffs_thresh, wavelet)
        flux_clean = flux_clean[:len(flux)]
        
        # ✅ KEY FIX: normalize AFTER denoising
        flux_clean = flux_clean / np.median(flux_clean)
        
        lc.flux = flux_clean * u.dimensionless_unscaled
    except Exception as e:
        # Fallback to simple normalization if wavelet fails
        lc = lc.normalize()
        
    # 3. Detrending (Flattening using Savitzky-Golay)
    try:
        # window_length must be odd, we use a large one so we don't remove transits
        lc_flat = lc.flatten(window_length=101)
    except Exception:
        lc_flat = lc
    
    return lc_flat

def extract_advanced_features(lc):
    """
    Enhanced Feature Engineering Pipeline (Updated for Rebuild).
    """
    time = np.array(lc.time.value, dtype=float)
    flux = np.array(lc.flux.value, dtype=float)
    
    if len(flux) < 100:
        return None
        
    f_mean = np.mean(flux)
    f_std = np.std(flux)
    f_min = np.min(flux)
    f_max = np.max(flux)
    f_range = f_max - f_min
    f_skew = float(pd.Series(flux).skew())
    f_kurt = float(pd.Series(flux).kurt())
    f_p5 = np.percentile(flux, 5)
    f_p95 = np.percentile(flux, 95)
    
    # BLS Transit Search
    try:
        model = BoxLeastSquares(time * u.day, flux)
        pg = model.autopower(0.1, minimum_period=0.5, maximum_period=15.0)
        best = np.argmax(pg.power)
        
        bls_power = float(pg.power[best])
        bls_period = float(pg.period[best].value)
        bls_duration = float(pg.duration[best].value)
        bls_depth = float(pg.depth[best]) if hasattr(pg, 'depth') else float(f_max - f_min)
        
        snr = bls_depth / f_std if f_std > 0 else 0.0
        
        mean_power = np.mean(pg.power)
        rel_power = bls_power / mean_power if mean_power > 0 else 0.0
    except Exception:
        bls_power = bls_period = bls_duration = bls_depth = snr = rel_power = 0.0
        
    return {
        'mean': f_mean, 'std': f_std, 'min': f_min, 'max': f_max, 'range': f_range,
        'skew': f_skew, 'kurt': f_kurt, 'p5': f_p5, 'p95': f_p95,
        'bls_power': bls_power, 'bls_period': bls_period,
        'bls_duration': bls_duration, 'bls_depth': bls_depth,
        'snr': snr, 'rel_power': rel_power
    }

def build_dataset(output_path="training_data.csv"):
    print("Building dataset...")
    # Expand the labeled stars list
    labeled_stars = [
        ('TIC 307210830', 1), ('TIC 25155310',  1), ('TIC 261136679', 1),
        ('TIC 149603524', 1), ('TIC 410214986', 1), ('TIC 167602025', 2), 
        ('TIC 229804573', 2), ('TIC 38846515',  2), ('TIC 272895692', 2),
        ('TIC 141306047', 3), ('TIC 294328691', 3), ('TIC 320697926', 3),
        ('TIC 92352620',  0), ('TIC 198456033', 0), ('TIC 350842851', 0),
        ('TIC 12345678',  0), ('TIC 149603525', 1), ('TIC 167602026', 2) # synthetic/additional
    ]
    
    features_list = []
    labels_list = []
    
    for tic_id, label in labeled_stars:
        start_time = time.time()
        try:
            print(f"Fetching {tic_id}...")
            sr = lk.search_lightcurve(tic_id, mission='TESS', exptime=120)
            if len(sr) == 0:
                sr = lk.search_lightcurve(tic_id, mission='TESS')
            if len(sr) == 0:
                print(f"  Skipped {tic_id}")
                continue
            
            lc = sr[0].download()
            if lc is None: continue
            
            lc_clean = preprocess_lightcurve(lc)
            feats = extract_advanced_features(lc_clean)
            
            if feats:
                features_list.append(feats)
                labels_list.append(label)
                proc_time = time.time() - start_time
                print(f"  Success: {tic_id} (Processed in {proc_time:.2f}s)")
                
                # Data Augmentation: Inject synthetic transit into noise targets (label 0)
                if label == 0:
                    print(f"  [Augmentation] Injecting synthetic transit into {tic_id}...")
                    # simple synthetic transit injection
                    synthetic_lc = lc_clean.copy()
                    t = synthetic_lc.time.value
                    period = np.random.uniform(2.0, 8.0)
                    t0 = t[0] + np.random.uniform(0, period)
                    duration = np.random.uniform(0.05, 0.2)
                    depth = np.random.uniform(0.005, 0.02)
                    
                    phase = ((t - t0) % period) / period
                    phase_width = duration / period
                    transit_mask = (phase < phase_width) | (phase > 1 - phase_width)
                    synthetic_lc.flux[transit_mask] *= (1.0 - depth)
                    
                    synth_feats = extract_advanced_features(synthetic_lc)
                    if synth_feats:
                        features_list.append(synth_feats)
                        labels_list.append(1) # Label 1 for Exoplanet
                        
        except Exception as e:
            print(f"  Failed: {tic_id} - {e}")
            continue
            
    df = pd.DataFrame(features_list)
    df['label'] = labels_list
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path} with {len(df)} samples.")
    return df

if __name__ == '__main__':
    build_dataset()
