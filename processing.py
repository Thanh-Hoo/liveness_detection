import pandas as pd
import numpy as np
from statistics import mean
from tqdm import tqdm

num_video_root = ""
df = pd.read_csv("result.csv")
df_final = []
for i in tqdm(range(len(df))):
    fname,class_,score = df['fname'][i], df['class'][i], df['score'][i]
    num_video, frame = fname.split("_")
    if num_video == num_video_root:
        if class_ == 1 :
            score +=1
            score_list.append(score)
        elif class_ == 0 :
            score = 1 - score
            score_list.append(score)
    else:
        if i != 0:
            liveness_score = mean(score_list)/3
            df_final.append([str(num_video_root)+".mp4",liveness_score])
        
        score_list = []
        num_video_root = num_video
        if class_ == 1 :
            score +=1
            score_list.append(score)
        elif class_ == 0 :
            score = 1 - score
            score_list.append(score)
liveness_score = mean(score_list)/3
df_final.append([str(num_video_root)+".mp4",liveness_score])
df_final = pd.DataFrame(df_final, columns=["fname","liveness_score"])
df_final.to_csv("Predict.csv", index =False)


    










