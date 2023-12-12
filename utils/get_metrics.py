from utils.orientdb import OrientDBClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from keybert import KeyBERT
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import squarify
import seaborn as sns
import numpy as np

class PaperAnalysis:
    def __init__(self):
        self.orientdb_client = OrientDBClient()
        self.orientdb_client.connect()
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=1800, max_df=0.95, min_df=2)
        #self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.num_clusters = 4
        self.get_total_papers_by_area()

    def get_papers_df(self):
        query = """SELECT @rid.asString() as id, ai_keywords
                   FROM Papers
                   WHERE processed = True
                   AND NOT (ai_keywords = '')"""
        result = self.orientdb_client.execute_query(query)
        list_keywords_papers = [record.ai_keywords for record in result]

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(list_keywords_papers)
        km = KMeans(n_clusters=self.num_clusters, random_state=42)
        km.fit(self.tfidf_matrix)
        self.clusters = km.labels_.tolist()

        self.papers_df = pd.DataFrame({'Paper': range(len(list_keywords_papers)), 'Keywords': list_keywords_papers, 'Cluster': self.clusters})
        return self.papers_df

    def get_total_from_query(self, query):
        result = self.orientdb_client.execute_query(query)
        if result:
            return result[0].total
        else:
            return 0

    def get_total_papers_by_area(self):
        self. total_by_area = {}
        for area in ['Life Sciences', 'Physical Sciences and Engineering',
                     'Health Sciences', 'Social Sciences and Humanities']:
            query = f"""SELECT count(*) as total FROM Papers WHERE study_area = '{area}'"""
            total_papers = self.get_total_from_query(query)
            self.total_by_area[area] = total_papers

    def get_total_papers(self):
        query = f"""SELECT count(*) as total FROM Papers"""
        return self.get_total_from_query(query)

    def get_cluster_counts(self):
        papers_df = self.get_papers_df()
        cluster_counts = papers_df.groupby('Cluster').count()
        return cluster_counts['Paper']

    def visualize_clusters(self, figsize=(10, 6)):
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.tfidf_matrix.toarray())

        # Convert clusters to a numpy array for correct indexing
        cluster_array = np.array(self.clusters)

        # Define a pastel color palette with as many colors as clusters
        n_clusters = len(set(self.clusters))
        pastel_colors = sns.color_palette("pastel", n_clusters)

        # Set the figure size based on the passed figsize parameter
        plt.figure(figsize=figsize)
        for cluster_label, color in zip(set(self.clusters), pastel_colors):
            cluster_points = X_pca[cluster_array == cluster_label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[color], marker='o', edgecolor='k', s=50)
        plt.title('Visualization of Paper Clusters (2D)')
        plt.xlabel('PCA Feature 1')
        plt.ylabel('PCA Feature 2')
        #plt.legend(title='Clusters')
        return plt


    from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting module

    def visualize_clusters_3d(self, figsize=(10, 10)):
        pca = PCA(n_components=3)  # Use 3 components for 3D visualization
        X_pca = pca.fit_transform(self.tfidf_matrix.toarray())

        # Convert clusters to a numpy array for correct indexing
        cluster_array = np.array(self.clusters)

        # Define a pastel color palette with as many colors as clusters
        n_clusters = len(set(self.clusters))
        pastel_colors = sns.color_palette("pastel", n_clusters)

        # Set the figure size and create a 3D subplot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')  # Add a 3D subplot

        print(f'{X_pca=}')
        for cluster_label, color in zip(set(self.clusters), pastel_colors):
            cluster_points = X_pca[cluster_array == cluster_label]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], c=[color], marker='o', edgecolor='k', s=50, label=f'Cluster {cluster_label}')
            print(f'{cluster_points[:, 0]=}')
            print(f'{cluster_points[:, 1]=}')
            print(f'{cluster_points[:, 2]=}')

        ax.set_title('3D Visualization of Paper Clusters')
        ax.set_xlabel('PCA Feature 1')
        ax.set_ylabel('PCA Feature 2')
        ax.set_zlabel('PCA Feature 3')
        plt.legend(title='Clusters')
        return plt


    def create_pie_chart(self, counts):
        labels = counts.index
        # Create pie chart
        plt.figure(figsize=(5, 5))
        plt.pie(counts, labels=labels, autopct='%1.1f%%', startangle=140,  radius=0.8)
        plt.title('Distribution of Papers Across Clusters')
        return plt

    def get_summary(self):
        cluster_dict = {}
        for cluster, keywords in self.papers_df.groupby('Cluster')['Keywords']:
            keyword_list = ', '.join(keywords).split(', ')
            cluster_dict[cluster] = keyword_list

        kw_extractor = KeyBERT('all-MiniLM-L6-v2')
        cluster_keywords_keybert = {}

        for cluster, group_df in self.papers_df.groupby('Cluster'):
            cluster_text = ' '.join(group_df['Keywords'])
            keywords = kw_extractor.extract_keywords(cluster_text, top_n=10)
            cluster_keywords_keybert[cluster] = {
                'keywords': [kw[0] for kw in keywords],
                'count': len(group_df)
            }
        return cluster_keywords_keybert

    def create_wordcloud(self, keywords_dict, cluster_number):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keywords_dict[cluster_number])
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'Word Cloud for Cluster {cluster_number}')
        return plt

    def create_bubble_chart(self):
        areas = list(self.total_by_area.keys())
        counts = list(self.total_by_area.values())
        plt.figure(figsize=(10, 6))
        plt.scatter(areas, [1]*len(areas), s=counts, alpha=0.5)
        plt.xlabel('Study Area')
        plt.ylabel('Constant (1)')
        plt.title('Bubble Chart of Papers by Area')
        return plt

    def create_tree_map(self):
        sizes = list(self.total_by_area.values())
        labels = list(self.total_by_area.keys())
        plt.figure(figsize=(12, 8))
        squarify.plot(sizes=sizes, label=labels, alpha=0.6)
        plt.title('Tree Map of Papers by Area')
        plt.axis('off')
        return plt

    def create_bar_chart(self):
        areas = list(self.total_by_area.keys())
        counts = list(self.total_by_area.values())
        sorted_areas, sorted_counts = zip(*sorted(zip(areas, counts), key=lambda x: x[1]))
        colors = sns.color_palette("pastel", len(areas))
        plt.figure(figsize=(10, len(areas) * 1.8))
        plt.barh(sorted_areas, sorted_counts, color=colors)
        plt.xlabel('Number of Papers')
        #plt.ylabel('Field of Knowledge')
        plt.title('Papers by Field of Knowledge')
        plt.gca().invert_yaxis()
        return plt