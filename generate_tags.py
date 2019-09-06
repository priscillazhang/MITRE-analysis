import gensim
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from sklearn.cluster import KMeans

from data_cleaning import *
from feature_extraction import *


# Store the most important keyword in each topic in a list and use this list to be the tags for that cluster
def GetTopics(corpus, vsm, num_topic):
    """
    Retrieve the topics and their corresponding keywords for the given corpus
    
    Parameters
    ----------
    corpus : list
        a list of lists of string tokens
    vsm : lists of tuples
        vector representation of words/phrases using a 
        specified wording encoding metric
    num_topic : integer
        number of topics summarized by lda
    
    Returns
    -------
    list of tuples
        returns topics and the keywords in each topic
        with their weights
    """

    dictionary = Dictionary(corpus)
    lda_model = gensim.models.LdaMulticore(
        vsm, num_topics=num_topic, id2word=dictionary, passes=2, workers=4)

    return lda_model.print_topics()


def ClusterTags(corpus, vsm, num_topic, topk, pattern):
    """
    Extract the most important k keywords in each topic
    from the given corpus. The topics are generated
    from LDA.
    
    Parameters
    ----------
    corpus : list
        a list of lists of string tokens
    vsm : lists of tuples
        vector representation of words/phrases using a 
        specified wording encoding metric
    num_topic : integer
        number of topics summarized by lda
    topk : integer
        number of keywords
    pattern : string
        the regex pattern of the desired text
        retrieved
    
    Returns
    -------
    list
        a list of top keywords from each topic
    """

    cluster_tag_lst = []
    topic_result = GetTopics(corpus, vsm, num_topic)
    for topic in topic_result:
        listsplit = str(topic[1].split('+')).split('*')
        for j in range(1, topk+1):
            keywords_clean = re.search(pattern, listsplit[j]).group()
            cluster_tag_lst.append(keywords_clean)

    return cluster_tag_lst


def remove_duplicates(x):
    """
    remove the duplicated words from
    the given list of words
    
    Parameters
    ----------
    x : list
        list of words 
    
    Returns
    -------
    list
        list of cleaned words
    """

    return list(dict.fromkeys(x))

# Create the master list that contains all the important topics in each cluster and the corpus
# Convert the format of cluster content so it matches the input form for lda modelling


def Cluster_format(cluster_content_list):
    """
    Format the words by grouping them
    into the cluster they belong to
    
    Parameters
    ----------
    cluster_content_list : list
        entire list of ungrouped corpus
    
    Returns
    -------
    list of words 
        list of words grouped by the cluster
        they belong to
    """

    cluster_list = []
    for clusters in cluster_content_list:
        description_list = []
        for description in clusters:
            description_corpus = ','.join(description)
            description_list.append(description_corpus)
        cluster_list.append(description_list)

    return cluster_list

# Find the list of topics in each cluster using BOW vsm on the data clustered using d2v metric

def TagGeneration(technique_corpus, keyword_list):
    """
    Generate tags by comparing 
    the corpus of each technique 
    description with the keywords obtained
    from topic modeling on the cluster
    which the technique belongs to. The keywords 
    appear in both the technique corpus and 
    topic modelling are the tags for that 
    technique
    
    Parameters
    ----------
    corpus : list
        list of words obtained from
        tokenizing the description of
        the technique 
    master_list : list 
        list of keywords extracted from
        the entire corpus using topic modeling 
        (lda)
    
    Returns
    -------
    list
        tags for that technique
    """
    technique_tags = list(
        set(technique_corpus).intersection(keyword_list))
    return technique_tags

def count_empties(lst, is_outer_list=True):
    """
    Count the number of empty entries 
    
    Parameters
    ----------
    lst : list of lists
        the input list that needed to be examined 
        if there are any empty entries
    is_outer_list : bool, optional
         by default True
    
    Returns
    -------
    integer 
        the number of empty entries
    """

    if lst == []:
        # the outer list does not counted if it's empty
        return 0 if is_outer_list else 1
    elif isinstance(lst, list):
        return sum(count_empties(item, False) for item in lst)
    else:
        return 0


def ClusterDescriptions(km, tech_df):
    """
    Returns the list of techniques in each cluster
    
    Parameters
    ----------
    km : class (sklearn.cluster.k_means_.KMeans)
        Cluster number each document belongs to
    tech_df : pandas dataframe
        the technique datframe created in the beginning
    
    Returns
    -------
    list
        the list of techniques in each cluster
    """

    cluster_content_list = []

    for i in set(km.labels_):
        cluster_corpus = [tech_df['word_list'].values[x].split(
            ',') for x in np.where(km.labels_ == i)[0]]
        cluster_content_list.append(cluster_corpus)

    return cluster_content_list


def kmeans(corpus, k, max_iteration=100, n_initial=5):
    """
    Run K Means algorithm 
    
    Parameters
    ----------
    corpus : Array-like, 2 dimensional
        Vector represenatation of the dataset.
    k : int
        Number of clusters
    max_iteration : int, optional
        Number of iterations K means algorithm runs for, by default 100
    n_initial : int, optional
        Number of times K Means will run with different centroid seeds, by default 5
    
    Returns
    -------
    list
        Cluster number each document belongs to
    """

    km = KMeans(n_clusters=k, init='k-means++', max_iter=max_iteration, n_init=n_initial,
                verbose=0)
    result = km.fit(corpus)

    return result

def main():

    data_clean = DataCleaning()
    feature = FeatureSelection()

    ##########################################################
    ######################## LOAD DATA #######################
    ##########################################################

    tech_df = pd.read_csv('tech_tactic.csv', index_col=0)

    ##########################################################
    ################# FEATURE ENGINEERING ####################
    ##########################################################

    word_list = data_clean.CorpusListWords(tech_df['Technique_description'])
    tech_df['word_list'] = pd.DataFrame(word_list)

    doc2vec_matrix = feature.Doc2Vec(word_list)
    bow_matrix = feature.Perform(word_list, 'bow')

    ##########################################################
    ######################## MODELLING #######################
    ##########################################################

    # Cluster the techniques
    km_d2v = kmeans(doc2vec_matrix, 12, max_iteration=100, n_initial=5)
    cluster_content = ClusterDescriptions(km_d2v, tech_df)

    #  Get topics for all the techniques
    pattern = r'[a-zA-Z\s]+'
    word_list_split=[doc.split(',') for doc in word_list]
    master_list_corpus = ClusterTags(
        word_list_split,bow_matrix,12,5,pattern)

    ### Start from here

    #  Get topics of techniques per cluster
    cluster_content_list_converted = Cluster_format(cluster_content)
    master_list_cluster = []
    for i in range(len(cluster_content_list_converted)):
        vsm_cluster = feature.Perform(
            cluster_content_list_converted[i], 'bow')
        cluster_content_list_split = [
            doc.split(',') for doc in cluster_content_list_converted[i]]
        cluster_tag = ClusterTags(
            cluster_content_list_split, vsm_cluster, 12, 5, pattern)
        master_list_cluster.append(cluster_tag)


    # Combine master_list_corpus and master_list_cluster and delete the duplicates to generate the final master list
    master_list_raw = master_list_corpus + master_list_cluster
    master_list_combine = []
    for lst in master_list_raw:
        master_list_combine += lst
    master_list = remove_duplicates(master_list_combine)

    #  Check if the tags to see if they exist in the descriptions
    #  TODO: Test if this is redundant.
    tag_list=[]
    for technique_description in word_list_split:
        tags=TagGeneration(technique_description,master_list)
        tag_list.append(tags)

    #  Remove duplicate duplicate tags
    tag_list_clean=[remove_duplicates(doc) for doc in tag_list]

    #  Format the tags for displaying in dataframe
    tags_df = [','.join(doc) for doc in tag_list_clean]

    #  Store the tags in a dataframe according to the technique they belong to
    df_technique = pd.DataFrame(tech_df['Technique_name'].values)
    df_technique.columns = ['Technique_name']
    df_technique['Tags'] = pd.DataFrame(tags_df)
    print(df_technique)
if __name__ == "__main__":
    main()
