import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

import argparse
# import warnings

class ClusteringAgent:
    def __init__(self, file_path, random_state=None):
        self.file_path = file_path
        self.data = None
        self.clusters = None
        self.n_clusters = None
        self.best_method = None
        self.random_state = None
        
    def load_data(self):
#        if type(self.file_path) != str:
#            self.data = self.file_path  #workaround to load the df.
#        else:
        try:
            self.data = pd.read_csv(self.file_path)
            return self.data
        except Exception as e:
            raise Exception(f"Error when laoding the data: {str(e)}")     
    
    def impute_missing(self):
        """This function imputes the missing values with the mean.
           In the future, it will contain a more comprehensive approach.
        """
        imputer = SimpleImputer(strategy='mean')
        numeric_cols = self.data.select_dtypes(include=np.number).columns  # Our example dataset has all numeric data, but...
        self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])

    def normalize(self):
        scaler = StandardScaler()
        numeric_cols = self.data.select_dtypes(include=np.number).columns
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])

    def preprocess_data(self):
        self.impute_missing()
        self.normalize()
        return self.data
    
    def determine_clusters(self, max_clusters=10, method='kmeans'):
        """ This function consider a clustering method (kmeans or agglomerative) and
            determines the optimal number of clusters via silhouette score.
            Future: score options."""
        best_score = -1
        best_n = 2
        
        for n in range(2, max_clusters+1):
            if method == 'kmeans':
                kmeans = KMeans(n_clusters=n, random_state=self.random_state)
                labels = kmeans.fit_predict(self.data.select_dtypes(include=np.number))
            else: #method == 'agglomerative'
                agglo = AgglomerativeClustering(n_clusters=n)
                labels = agglo.fit_predict(self.data.select_dtypes(include=np.number))

            score = silhouette_score(self.data.select_dtypes(include=np.number), labels)
            
            if score > best_score:
                best_score = score
                best_n = n
                
        self.n_clusters = best_n
        return best_n
    
    def cluster_data(self):
        """Run the best possible clustering method."""
        
        # Only numeric data (in our case it is redundant)
        # Future: take in account categorical data if is needed.
        numeric_data = self.data.select_dtypes(include=np.number)

        methods = {'KMeans': KMeans(n_clusters=self.n_clusters, random_state=self.random_state),
                   'Agglomerative': AgglomerativeClustering(n_clusters=self.n_clusters),
                   'DBSCAN': DBSCAN(eps=0.5, min_samples=5)}  # Using default Hyperparameters for simplicity.

        best_score = -1
        best_labels = None
        best_method = None

        for name, method in methods.items():
            try:
                labels = method.fit_predict(numeric_data)

                if len(set(labels)) < 2: # In case no cluster found (DBSCAN)
                    continue
                    
                score = silhouette_score(numeric_data, labels)
                
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    best_method = name
            except:
                continue
                
        self.best_method = best_method  # Method name
        self.data['cluster'] = best_labels  # Label (cluster) column
        self.clusters = best_labels
        
        return self.data
    
    def visualize_clusters(self, show=True, save=False):
        """Visualize clusters if data is 2D or can be reduced to 2D/3D"""
        numeric_data = self.data.select_dtypes(include=np.number).drop('cluster', axis=1, errors='ignore')
        
        # Reduce dimensions if needed
        if len(numeric_data.columns) > 3:
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(numeric_data)
            x, y = reduced_data[:, 0], reduced_data[:, 1]
            title_suffix = " (with PCA)"
        elif len(numeric_data.columns) == 3:
            x, y, z = numeric_data.iloc[:, 0], numeric_data.iloc[:, 1], numeric_data.iloc[:, 2]
            title_suffix = ""
        else:
            x, y = numeric_data.iloc[:, 0], numeric_data.iloc[:, 1]
            title_suffix = ""
        
        # Plot
        plt.figure(figsize=(5, 3))
        if len(numeric_data.columns) == 3:
            ax = plt.axes(projection='3d')
            ax.scatter(x, y, z, c=self.clusters, cmap='viridis')
            ax.set_title(f'3D Visualization - {self.best_method}{title_suffix}')
        else:
            plt.scatter(x, y, c=self.clusters, cmap='viridis')
            plt.title(f'Cluster Visualization - {self.best_method}{title_suffix}')

        if save:
            plt.savefig('figure_output.png')
        if show:
            plt.show()

    def save_results(self, output_file='output.csv'):
        # Save the original data with the cluter column
        # Warning: It needs to be improved in case of a more complex preprocessing (like row deleting)
        original_data = pd.read_csv(self.file_path)
        original_data['cluster'] = self.data['cluster']
        original_data.to_csv(output_file, index=False)

        # Save the processed data with the cluster column
        #self.data.to_csv(output_file, index=False)
        return output_file
    
    def run_pipeline(self, verbose=True, show_figure=True, save_figure=True):
        """Execute the full pipeline"""
        self.load_data()
        self.preprocess_data()
        self.determine_clusters()
        self.cluster_data()

        if verbose:
            print(f"Clustering completed using {self.best_method} with {self.n_clusters} clusters")
            print(f"Silhouette Score: {silhouette_score(self.data.select_dtypes(include=np.number), self.clusters):.2f}")
        
        try:
            self.visualize_clusters(show=show_figure, save=save_figure)
        except Exception as e:
            print(f"Visualization error: {str(e)}")
        
        output_path = self.save_results()
        print(f"Results saved to {output_path}")
        return self.data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autonomous Clustering Agent')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--output', help='Output file name (csv).', type=str, default='output.csv')
    parser.add_argument('--seed', help='Random state seed.', type=int, default=1234)
    parser.add_argument('--show-plot', help='Shoe plot.', action='store_true')
    parser.add_argument('--save-plot', help='Save plot.', action='store_true')

    args = parser.parse_args()

    cluster_agent = ClusteringAgent(args.input_file)
    clustered_data = cluster_agent.run_pipeline(show_figure=args.show_plot, save_figure=args.save_plot,)
    cluster_agent.save_results(args.output)
