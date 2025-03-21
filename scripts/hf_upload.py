from huggingface_hub import HfAPi
api = HfApi()

api.upload_folder(                                                                                                                                  
    folder_path="/home/tliu/learning-ivs/datasets/linear/lennon100-range-tau-50k/train3",                                                                
    repo_type="dataset",                                                                                                                               
    path_in_repo="train3/",                                                                                                                              
    repo_id="learning-ivs/lennon100-range-tau-50k",                                                                                                                                                                                                                      
    token=True)  