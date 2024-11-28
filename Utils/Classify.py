import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.interpolate import interp1d, UnivariateSpline
import matplotlib.pyplot as plt


class Classify:
    def __init__(self, all_msd:list[np.array]) :
        self.all_msd = all_msd

    def slope_classify(self,):
        slopes = []
        for msd in self.all_msd:
            time_lags = msd[0, :]  # First row: time lags
            msd_values = msd[1, :] # Second row: corresponding MSD values
            time_lags = np.log(time_lags)
            msd_values = np.log(msd_values)

            # Perform a linear fit: slope and intercept
            slope,_ = np.polyfit(time_lags, msd_values, 1)  # 1 for linear fit
            slopes.append(slope)

        # Convert slopes to NumPy array for processing
        slopes = np.array(slopes)

        # Plot a histogram of slopes
        plt.hist(slopes, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Slope')
        plt.ylabel('Frequency')
        plt.title('Histogram of MSD Slopes')
        plt.show()



    def kmean_arrays(self, arrays, n_clusters=None):
        normalized_features = self.process_msd(arrays)

        if n_clusters is None:
            n_clusters, _ = self.find_optimal_clusters(normalized_features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(normalized_features)
        
        return labels, kmeans, normalized_features

    @staticmethod
    def process_msd(all_msd:list[np.array]) :
        max_length = max([arr.shape[1] for arr in all_msd])
    
        # Interpolate all arrays to have the same length
        processed_data = []
        for arr in all_msd:
            time_lags = np.log(arr[0])
            msd_values = np.log(arr[1])
            # Create interpolation function
            if len(time_lags) <= 3:  # Use linear interpolation for small datasets
                f = interp1d(time_lags, msd_values, kind='linear', fill_value='extrapolate')
            else:  
                f = UnivariateSpline(time_lags, msd_values, s=0)

            # Create uniform time points
            uniform_time = np.linspace(min(time_lags), max(time_lags), max_length)
            # Interpolate MSD values
            uniform_msd = f(uniform_time)
            processed_data.append(uniform_msd)
    
        # Convert to numpy array
        features = np.array(processed_data)
    
        # Normalize features
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features)
        
        return normalized_features

    def find_optimal_clusters(self, features, max_clusters=10):
        """
        Find optimal number of clusters using silhouette score.
        
        Parameters:
        features (np.array): Preprocessed features
        max_clusters (int): Maximum number of clusters to try
        
        Returns:
        optimal_k (int): Optimal number of clusters
        scores (list): List of silhouette scores
        """
        scores = []
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(features)
            score = silhouette_score(features, labels)
            scores.append(score)
        
        optimal_k = np.argmax(scores) + 2
        return optimal_k, scores


