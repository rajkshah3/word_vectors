import spacy
import numpy as np 
from numpy import unravel_index

nlp = spacy.load('en_core_web_lg')

def sentance_similarity(s1,s2,nlp):
    s1 = nlp(s1)
    s2 = nlp(s2)
    s1 = [tok.vector for tok in s1]
    s2 = [tok.vector for tok in s2]
    s1_mat = np.matrix(s1)
    s2_mat = np.matrix(s2)
    s1_mat = s1_mat / np.linalg.norm(s1_mat, axis=-1)[:, np.newaxis]
    s2_mat = s2_mat / np.linalg.norm(s2_mat, axis=-1)[:, np.newaxis]
    print(np.shape(s1_mat))
    sim_matrix = s1_mat*s2_mat.T
    print(np.shape(sim_matrix))
    similarity_metric = 0
    for i in range(min(len(s1),len(s2))):
        maxs = unravel_index(sim_matrix.argmax(), sim_matrix.shape)
        similarity_metric += sim_matrix[maxs[0],maxs[1]]
        print(similarity_metric)
        sim_matrix[maxs[0],:] = 0
        sim_matrix[:,maxs[1]] = 0
    print(similarity_metric)
    similarity_metric = similarity_metric/(np.sqrt(len(s1)*len(s2)))
    print(similarity_metric)
    return similarity_metric

s1 = 'here is some text'
s2 = 'there is some more text here'
sentance_similarity(s1,s2,nlp)
