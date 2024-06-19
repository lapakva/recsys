import pandas as pd
import numpy as np

model_matrix = pd.read_parquet('model_matrix.pq')


def cosine_sim_vector(vector, base, top_n=10):
    # Calculating the dot product using matrix multiplication
    dot_product_matrix = np.dot(base, vector.T).flatten()
    max_indices = np.argsort(dot_product_matrix)[-top_n:][::-1]   
    max_values = dot_product_matrix[max_indices]
    return max_indices, max_values

def cosine_sim_matrix(matrix, base, top_n=10):
    """this function calculate cosine similarity between all talents and job,
        by using numpy for matrix multiplication,
        to speed up can be mobified to use pyTorch on GPU(get x10-x100in speed)
        In casse of RAM issuse can be executed in butches"""
    dot_product_matrix = np.dot(matrix, base.T) 
    max_indices = np.argsort(dot_product_matrix, axis=1)[:, -top_n:][:, ::-1]
    max_values = np.take_along_axis(dot_product_matrix, max_indices, axis=1)
    print(max_indices, max_values)
    return max_indices, max_values
    

class Search:
    def __init__(self, model_matrix=model_matrix) -> None:
        self.model_matrix = model_matrix

    def match(self, talent_id: int, job_id: int) -> dict:
        # ==> Method description <==
        #As I already prepare embadding for talends and jobs, and save it in file, it will be easier
        # and faster to get this embedding by id
        # This method takes a talent_id and job_id as input and uses the machine learning
        # model to predict the label. 
        # To go derectly to recommendation system method will find 10 closest jobs_id for one telant_id,
        # if list of recommendation for telent_id and job_ids are overlaping
        #method print 'Match' as True, and a cosine similarity for them  

        talent_vector = self.model_matrix[self.model_matrix.talent_id == talent_id].iloc[:,:-5].values
        # print("talent_vector", talent_vector)

        base = self.model_matrix[self.model_matrix.talent_or_job == 'job'].iloc[:,:-5].values
        max_indices, max_values = cosine_sim_vector(talent_vector, base) 
        print(max_indices, max_values)
        matched_index = np.where(max_indices == job_id)[0]

        if matched_index.size > 0:
            matched_index = matched_index[0]
            print(f"Matched index: {matched_index}")            
        else:
            print(f"Job ID {job_id} not found in top {len(max_indices)} indices")
                

        
        

    def match_bulk(self, talent_ids: list, job_id: list):
        # ==> Method description <==
        ##As I already prepare embadding for talends and jobs, and save it in file, it will be easier
        # and faster to get this embedding by id
        # This method takes a talent_id and job_id as input and uses the machine learning
        # model to predict the label. 
        # To go derectly to recommendation system method will find 10 closest jobs_id for all telant_id in the list,
        # if list of recommendation for telent_id and job_ids are overlaping
        #method print 'Match' as True, and a cosine similarity for them  


        talent_matrix = self.model_matrix[self.model_matrix.talent_id.isin(talent_ids)].iloc[:,:-5].values
        print("talent_vectors", talent_matrix)
        base = self.model_matrix[self.model_matrix.talent_or_job == 'job'].iloc[:,:-5].values
        print("base", base)
        max_indices, max_values = cosine_sim_matrix(talent_matrix, base)
        print("max_indices, max_values", max_indices, max_values)

        # Find matching job index for each talent id
        #this can be better parallelised(here cycle for visualisation perpases ) 

        for i, talent_id in enumerate(talent_ids):
                    print(max_indices[i])
                    matched_indices = np.where(max_indices[i] == job_id)[0]
                    
                    if matched_indices.size > 0:
                        matched_index = matched_indices[0]  # Take the first matched index
                        
                        print(f"Talent ID {talent_id}: Matched index: {matched_index}")
                    else:
                        print(f"Talent ID {talent_id}: Job ID {job_id} not found in top {len(max_indices[i])} indices")

        


if __name__=="__main__":
    search = Search()
    # search.match(1,1)
    search.match_bulk([1,2,3],1)
